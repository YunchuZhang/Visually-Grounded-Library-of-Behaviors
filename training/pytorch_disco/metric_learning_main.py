import os
import copy
import torch
import random
import argparse
import datetime
import matplotlib
import numpy as np
matplotlib.use('Agg')
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from nets.quant_policy_metric_models import EmbeddingGenerator1D, EmbeddingGenerator3D
from backend.quant_metric_inputs import MetricLearningData, ValidationLoader
from nets.quant_policy_metric_models import compute_loss1D, compute_loss3D

# set the seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_objects_name(train_files):
    """
        returns: object name, unique str in case of shapenet
        given names in case of pallet
    """
    objs = list()
    for t in train_files:
        splits = t.split('/')[-1]
        obj = splits[:-4]
        objs.append(obj)
    return objs

def get_rgb_paths(obj_names, labels):
    """
        returns a (list) of rgb_paths corresponding to the objects
        This is path to entire visual data
    """
    base_dirs = "/home/ubuntu/quant_data/controllers_data"  # TODO:TODO: Need to remove hardcoding
    rgb_paths = list()
    for o, l in zip(obj_names, labels):
        if l == 0:
            # means object belongs to class 0
            rgb_path = os.path.join(base_dirs, 'controller1_data',
                f'visual_data_{o}.npy')
            assert os.path.exists(rgb_path)
        if l == 1:
            # means the object belongs to class 1
            rgb_path = os.path.join(base_dirs, 'controller2_data',
                f'visual_data_{o}.npy')
            assert os.path.exists(rgb_path)
        rgb_paths.append(rgb_path)
    return rgb_paths

def plot_matplotlib(vis, log_path, step, dist_fn):
    """
        vis: np.ndarray (containing len(val_files) rows and k columns)
        log_path: path to log_dir of this experiment
        step: which step of the training am I in
        dist_fn: which distance function was used to compute nearest neighbours 
    """
    n_rows = len(vis)
    n_cols = len(vis[0])
    fig_size = 2 * np.asarray([20,10])
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size,
        sharex=True, sharey=True)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i][j].imshow(vis[i][j])
    
    plt.tight_layout()
    # save the figure and you are done
    fig.savefig(f'{log_path}/nn_result_{dist_fn}_{step}.jpg')
    plt.close()

def plot_grid(vis, log_path, step, dist_fn):
    """
        vis: np.ndarray (containing len(val_files) rows and k columns)
        log_path: path to log_dir of this experiment
        step: which step of the training am I in
        dist_fn: which distance function was used to compute nearest neighbours
        returns:
        --------
            a grid containing query object and its nearest neighbours
    """
    H, W, C = list(vis[0][1].shape)
    vis = np.stack(vis, axis=0)
    # convert to torch, Rows, cols, chans, height, width
    vis_nearest = torch.from_numpy(vis).permute(0, 1, 4, 2, 3)
    # resize
    vis_nearest = vis_nearest.view(-1, C, H, W)
    # make the grid
    grid = make_grid(vis_nearest, nrow=len(vis[0]))
    return grid
    

def plot_nearest_neighbours(dist_mat, train_files, val_files, train_labels, val_labels,
    log_path, step, dist_fn, plot_method='grid'):
    """
        dist_mat: (torch.topk tuple)
            is a tuple of values, index each of them have len(val_files_rows)
        train_files: (list)
            files used for training, based on which I will get the name of obj
        val_files: (list)
            files used for validation, same func as above
        train_labels: (list)
            to locate the path
        val_labels: (list)
            to locate the path
        dist_fn: (str)
            whether the distance used is dot or eucl
        plot_method: (str)
            whether to use matplotlib plotting or torch style grid plot
        
        returns:
        -------------
            if matplotlib is used as plotting method it returns None
            if grid is used as plotting methods it returns the grid
    """
    train_objects = get_objects_name(train_files)
    val_objects = get_objects_name(val_files)

    train_rgb_paths = get_rgb_paths(train_objects, train_labels)
    val_rgb_paths = get_rgb_paths(val_objects, val_labels)

    values, indices = dist_mat

    V, T = list(indices.shape)
    vis = list()  # holds the results

    cnt = 0
    for v in range(V):
        minivis = list()
        val_f = np.load(val_rgb_paths[v], allow_pickle=True).item()
        val_ref_im = val_f['rgb_camRs'][0]
        minivis.append(val_ref_im)
        # label of the current val image
        curr_label = val_labels[v]
        for t in range(T):
            t_idx = indices[v][t]
            # label of the current train image
            curr_tlabel = train_labels[t_idx]
            if curr_label == curr_tlabel and t == 0:
                cnt += 1
            train_f = np.load(train_rgb_paths[t_idx], allow_pickle=True).item()
            train_ref_im = train_f['rgb_camRs'][0]
            minivis.append(train_ref_im)
        vis.append(minivis)
    
    precision_at_recall1 = float(cnt) / len(indices) * 100
    print(f'precision at recall 1 is {precision_at_recall1}')

    if plot_method == 'matplotlib':
        plot_matplotlib(vis, log_path=log_path, step=step, dist_fn=dist_fn)
        return None
    elif plot_method == 'grid':
        grid = plot_grid(vis, log_path=log_path, step=step, dist_fn=dist_fn)
        return grid

def compute_distance3Dfast(train_embs, val_embs, dist_fn='eucl',
    log_path=None, step=None):
    B, C, D, H, W = list(train_embs.shape)
    B1, C, D, H, W = list(val_embs.shape)
    train_embs = train_embs.view(B,-1)
    val_embs = val_embs.view(B1, -1)
    # now use the cdist to compute the distance and check if its same
    dist_mat = torch.cdist(val_embs, train_embs)
    topk = torch.topk(dist_mat, k=5, dim=1, largest=False, sorted=True)
    return topk

def compute_distance3D(train_embs, val_embs, dist_fn='eucl',
    log_path=None, step=None):
    """All the args are same as below, but dist is computed using
       euclidean norm for each val with every train embs."""
    
    major_dist = list()
    for i, v in enumerate(val_embs):
        curr_v = v
        minidist = list()
        for j, t in enumerate(train_embs):
            curr_t = t
            dist = torch.norm(v-t)
            minidist.append(dist)
        major_dist.append(minidist)
    
    for i, d in enumerate(major_dist):
        major_dist[i] = torch.Tensor(d).float().to(train_embs.device)
    
    dist_mat = torch.stack(major_dist)
    topk = torch.topk(dist_mat, k=5, dim=1, largest=False, sorted=True)
    return dist_mat, topk
 
def compute_distance(train_embeddings, val_embeddings, dist_fn='both',
    log_path=None, step=None):
    """
        train_embeddings: (len(train_embeddings), emb_dim) torch.Tensor
        val_embeddings: (len(val_embeddings, emb_dim)) torch.Tensor
        dist_fn: ['both', 'eucl', 'dot']
        log_path: log path of current experiment
        step: current training step

        returns:
        ----------
            if dist_fn == 'eucl', top_k nn for val objects from train object
                computed using euclidean norm
            if dist_fn == 'dot', top_k nn for val objects from train objects
                computed using dot product
            if dist_fn == 'both', top_k are computed using both and list is returned
    """
    assert dist_fn in ['dot', 'eucl', 'both'], "Ain't gonna happen, you need\
        to comply to me son"
    if dist_fn == 'eucl' or dist_fn == 'both':
        dist_mat_eucl = torch.cdist(val_embeddings, train_embeddings)
        # smallest distance should be at the top for euclidean distance
        eucl_top_k = torch.topk(dist_mat_eucl, k=5, dim=1, largest=False, sorted=True)
    
    if dist_fn == 'dot' or dist_fn == 'both':
        # now do the similar things for dot product as well
        dist_mat_dot = torch.mm(val_embeddings, train_embeddings.t())
        # largest distance should be at the top for the dot
        dot_top_k = torch.topk(dist_mat_dot, k=5, dim=1, largest=True, sorted=True)
    
    if dist_fn == 'eucl':
        return eucl_top_k
    elif dist_fn == 'dot':
        return dot_top_k
    elif dist_fn == 'both':
        return [eucl_top_k, dot_top_k]

def validate_(train_file, val_file, model, device, dist_fn, writer, loss_fn, log_path, step,
    plot_method='grid'):
    """
        Now this is going to be costly function
        I will forward pass through all the train_data
        I will forward pass through all the val_data
        for each of the val_data point I will compute its distance
        (euclidean, dot) for now from each of the each of train_data
        collect the top-10 nearest(ascending in case of euclidean) and
        (descending in case of dot) neighbours and display them in grid.

        parameters
        -----------------
        train_file      : (str) dataloader for train_data
        val_file        : (str) dataloader for val_data
        model           : (torch.nn.Module) forward pass model
        device          : (torch.device) 'cuda' or 'cpu'
        dist_fn         : (str) dot, eucl, both
        writer          : (SummaryWriter)
        loss_fn         : (quant_policy_metric_models)
        log_path        : (str) what is the log path
        step            : (int) epoch_number
        plot_method     : (whether) to use grid plotting (torch style) or matplotlib

        returns:
        -----------------
            if plot_method == matplotlib and dist_fn == 'both':
                return [None, None, dist_fn]
            if plot_method == grid and dist_fn == 'both':
                return [grid_eucl, grid_dot, dist_fn]
            other combinations make sense from above pattern
    """
    # Step1: put the model in eval mode
    model.eval()
    assert model.training == False, "should be in eval mode"
    
    # Step 2: get the train and the val data 
    train_data_loader = ValidationLoader(train_file)
    train_iterator = train_data_loader.yield_data()
    
    val_data_loader = ValidationLoader(val_file)
    val_iterator = val_data_loader.yield_data()

    # Step 3: do the train data forward pass
    train_embeddings = list()
    train_files = list()
    train_class = list()
    for i, data in enumerate(train_iterator):
        ob_tensor = data['ob_tensor']
        train_class.append(data['label'])
        train_files.append(data['file_name'])

        # prepare for the forward pass
        ob_tensor = ob_tensor.to(device)
        # do the forward pass
        with torch.no_grad():
            output = model(ob_tensor)
        train_embeddings.append(output)
    
    train_embeddings = torch.cat(train_embeddings, dim=0)
    assert train_embeddings.shape[0] == len(train_data_loader)
    
    # Step 4: do the val data forward pass
    val_embeddings = list()
    val_files = list()
    val_class = list()

    for i, vdata in enumerate(val_iterator):
        ob_tensor = vdata['ob_tensor'].to(device)
        val_class.append(vdata['label'])
        val_files.append(vdata['file_name'])

        # prepare for the forward pass
        ob_tensor = ob_tensor.to(device)
        # do the forward pass
        with torch.no_grad():
            output = model(ob_tensor)
        val_embeddings.append(output)
    
    val_embeddings = torch.cat(val_embeddings, dim=0)
    assert val_embeddings.shape[0] == len(val_data_loader)

    # Step 5: do the nearest neighbour distance computation
    if val_embeddings.ndim > 2:
        # meaning I am using 3d part of the network
        # dist_matrix, topk = compute_distance3D(train_embeddings, val_embeddings,
        #     dist_fn=dist_fn, log_path=log_path, step=step)
        dist_matrix = compute_distance3Dfast(train_embeddings, val_embeddings,
            dist_fn=dist_fn, log_path=log_path, step=step)
    else:
        dist_matrix = compute_distance(train_embeddings, val_embeddings, dist_fn=dist_fn,
            log_path=log_path, step=step)
    
    # Step 6: plot the nearest neigbours #TODO: This kind of looks ugly make it pretty
    if isinstance(dist_matrix, list):
        grid_eucl = plot_nearest_neighbours(dist_matrix[0], train_files, val_files, train_class,
            val_class, log_path=log_path, step=step, dist_fn='eucl', plot_method=plot_method)
        grid_dot = plot_nearest_neighbours(dist_matrix[1], train_files, val_files, train_class,
            val_class, log_path=log_path, step=step, dist_fn='dot', plot_method=plot_method)
        return [grid_eucl, grid_dot, dist_fn]
    else:
        grid = plot_nearest_neighbours(dist_matrix, train_files, val_files, train_class,
            val_class, log_path=log_path, step=step, dist_fn=dist_fn, plot_method=plot_method)
        return [grid, dist_fn]


def train_(model, optimizer, dataloader, loss_fn, device):
    model.train()
    assert model.training == True, "should be in train mode"
    total_loss = 0.0
    iter_cnt = 0
    for i, data_dict in enumerate(dataloader):
        pos_tensor = data_dict['pos_tensor'].to(device)
        neg_tensor = data_dict['neg_tensor'].to(device)

        pos_labels = data_dict['pos_label'].to(device).long()
        neg_labels = data_dict['neg_label'].to(device).long()

        ob_tensors = torch.cat([pos_tensor, neg_tensor], dim=0)
        ob_labels = torch.cat([pos_labels, neg_labels], dim=0)

        # do the forward pass through the model
        output = model(ob_tensors)
        # compute the loss
        loss = loss_fn(output, ob_labels, device=device)

        # do the optimization
        p_before = [copy.deepcopy(p) for p in model.parameters()]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        p_after = [copy.deepcopy(p) for p in model.parameters()]
        truth_val = [torch.equal(a, b) for a, b in zip(p_before, p_after)]
        # we want the values to change so all the truth_values should be false
        assert not all(truth_val), "some weights are not changing"
        
        total_loss += loss.cpu().item()
        iter_cnt += 1
    
    avg_loss = total_loss / float(iter_cnt)
    return avg_loss


def main():
    parser = argparse.ArgumentParser('ArgumentParser for the metric learning')

    parser.add_argument('--model_type', type=str, default='3D',
        help='do we want 3d model or 2d model')
    parser.add_argument('--emb_dim', type=int, default=16,
        help='latent embedding dimension')
    
    ### ......... Data Args ............ ###
    parser.add_argument('--data_dir', type=str, default='backend/quant_train_files',
        help='base data dir where the files are located')
    parser.add_argument('--data_file', type=str, required=True,
        help='where is the data my man!!!')
    parser.add_argument('--val_data_file', type=str, default=None,
        help='where is the validation data file')
    
    ### ......... SavePath ........... ####
    parser.add_argument('--model_save_path', type=str, default="quant_ckpts",
        help='path where to save model checkpoints')
    parser.add_argument('--log_path', type=str, default="quant_logs",
        help='path to save logs of metric learning experiments')
    
    #### ........ Training parameters .......... ####
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_after', type=int, default=50)
    parser.add_argument('--log_after', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--do_val', action='store_true', default=False)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--add_lr_scheduler', action='store_true', default=False)
    ########## .... LR Scheduler parameters ........ ####
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=int, default=0.1)

    #### ....... Plotting parameters ........... ####
    parser.add_argument('--plot_method', type=str, default='grid',
        help='can be either matplotlib or grid')
    
    args = parser.parse_args()

    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    # get time too for generating unique elements
    dtime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_name = f"mtype_{args.model_type}_emb_{args.emb_dim}_df_{args.data_file[:-4]}_time_{dtime}"
    log_path = f"{args.log_path}/{exp_name}"
    ckpt_path = f"{args.model_save_path}/{exp_name}"
    train_file_path = f"{os.getcwd()}/{args.data_dir}/{args.data_file}"
    val_file_path = f"{os.getcwd()}/{args.data_dir}/{args.val_data_file}"
    print(train_file_path)
    print(val_file_path)

    print(exp_name)
    print(log_path)
    print(ckpt_path)

    assert os.path.exists(train_file_path), "what is the train file path"
    if args.do_val:
        assert os.path.exists(val_file_path)

    # Step 1: Generate the dataset #
    train_dataset = MetricLearningData(dataset_txt=train_file_path, transforms=None)
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Step 2: make the directories
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    # set up tensorboard, at the log_path
    writer = SummaryWriter(log_path)

    # Step 3: make the model and the optimizer
    if args.model_type == '3D':
        model = EmbeddingGenerator3D(args.emb_dim).to(device)
        loss_fn = compute_loss3D
    elif args.model_type == '1D':
        model = EmbeddingGenerator1D(args.emb_dim).to(device)
        loss_fn = compute_loss1D
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.add_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )


    # Step 4: train the model
    # copy the weights of the model here once, and compare it with the weights of
    # the model after returning from the above optimization
    pre_opt_weights = [p.clone() for p in model.parameters()]
    for n_ep in range(args.n_epochs):
        avg_epoch_loss = train_(model, optimizer, trainDataLoader, loss_fn, device=device)
        print(f'Epoch: {n_ep}\t Loss: {avg_epoch_loss}')
        writer.add_scalar('train_loss', avg_epoch_loss, n_ep)

        if n_ep % args.save_after == 0:
            model_path = f'{ckpt_path}/model_{n_ep}.pth'
            torch.save(model.state_dict(), model_path)
        
        if n_ep % args.log_after == 0:
            if args.do_val:
                return_vals = validate_(train_file_path, val_file_path, model, device=device,
                    dist_fn='eucl', writer=writer, loss_fn=loss_fn, log_path=log_path,
                    step=n_ep, plot_method=args.plot_method)
                
                if args.plot_method == 'grid':
                    if len(return_vals) == 3:
                        grid_eucl, grid_dot, dist_fn = return_vals
                        writer.add_image('nn_eucl', grid_eucl, n_ep)
                        writer.add_image('nn_dot', grid_dot, n_ep)
                    elif len(return_vals) == 2:
                        grid, dist_fn = return_vals
                        writer.add_image(f'nn_{dist_fn}', grid, n_ep)

        # change the scheduler stuff 
        if args.add_lr_scheduler:
            scheduler.step()
    
    # now finally when all of this over save all the parameters you used for this experiment
    # in its log folder
    with open(f'{log_path}/exp_params.txt', 'w') as f:
        dict_args = vars(args)
        for k, v in dict_args.items():
            f.write(f'{k} = {v}\n')
    f.close()
    

if __name__ == '__main__':
    main()