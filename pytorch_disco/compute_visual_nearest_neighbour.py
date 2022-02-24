import os
import torch
import argparse
import numpy as np

from backend.quant_metric_inputs import ValidationLoader
from metric_learning_main import plot_nearest_neighbours

def compute_dist_naive(emb_train, emb_val):
    """
        emb_train: NTrain, nlocs, emb_dim
        emb_val: NVal, nlocs, emb_dim

        returns:
        -----------
            dist_mat: distance of each emb_val from every emb_train
            So it would be (NVal, NTrain)
    """
    major_dist = list()
    for i, v in enumerate(emb_val):
        print(f'working on {i}/{len(emb_val)} val')
        curr_v = v
        minidist = list()
        for j, t in enumerate(emb_train):
            print(f'working on {j}/{len(emb_train)} train')
            curr_t = t
            dist = torch.norm(v.contiguous().view(-1)\
                - t.contiguous().view(-1))
            minidist.append(dist)
        major_dist.append(minidist)
    for i, d in enumerate(major_dist):
        major_dist[i] = torch.Tensor(d).float()
    
    dist_mat = torch.stack(major_dist)
    return dist_mat

def compute_nearest_neighbours(train_data, val_data, save_path):
    """
    parameters
    -------------
        train_data: (quant_metric_inputs.ValidataLoader)
        val_data: (quant_metric_inputs.ValidationLoader)
        save_path: where the plot will be stored
    """
    # gather just the tensors
    train_embeddings = list()
    train_objs = list()
    train_labels = list()
    train_files = list()
    for t in train_data.records:
        train_embeddings.append(t['ob_tensor'].squeeze(0))
        train_objs.append(t['ob_name'])
        train_labels.append(t['label'])
        train_files.append(t['file_name'])
    
    train_embeddings = np.stack(train_embeddings, axis=0)

    val_embeddings = list()
    val_objs = list()
    val_labels = list()
    val_files = list()
    for v in val_data.records:
        val_embeddings.append(v['ob_tensor'].squeeze(0))
        val_objs.append(v['ob_name'])
        val_labels.append(v['label'])
        val_files.append(v['file_name'])
    
    val_embeddings = np.stack(val_embeddings, axis=0)

    # some checks
    assert len(val_files) == len(val_embeddings), "should be equal bro"
    assert len(train_files) == len(train_embeddings), "should be equal brother"

    emb_train = torch.from_numpy(train_embeddings).float()
    emb_val = torch.from_numpy(val_embeddings).float()

    trainN, C, D, H, W = list(emb_train.shape)
    valN, _, _, _, _ = list(emb_val.shape)
    emb_train = emb_train.permute(0, 2, 3, 4, 1).reshape(trainN, D*H*W, C)
    emb_val = emb_val.permute(0, 2, 3, 4, 1).reshape(valN, D*H*W, C)

    dist_mat = compute_dist_naive(emb_train, emb_val)

    # now compute the top_k along axis = 1 and then plot it, I have computed the
    # euclidean distance remember that, so the smallest is required so largest should
    # be false, can easily verify using values returned to me if they are ascending or
    # descending, since the values themselves are sorted
    dist_mat_topk = torch.topk(dist_mat, k=5, dim=1, largest=False, sorted=True)
    plot_nearest_neighbours(dist_mat_topk, train_files, val_files, train_labels, val_labels,
        log_path=save_path, step=0, dist_fn='eucl', plot_method='matplotlib')


def main():
    parser = argparse.ArgumentParser('Parser for rgb-viewpred nn')
    parser.add_argument('--dataset_listdir', type=str,
        default='/home/ubuntu/pytorch_disco/backend/quant_train_files')
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    train3d_tensors_path = os.path.join(args.dataset_listdir, args.train_file)
    val3d_tensors_path = os.path.join(args.dataset_listdir, args.val_file)

    if not os.path.exists(train3d_tensors_path):
        raise FileNotFoundError('could not find training file')
    if not os.path.exists(val3d_tensors_path):
        raise FileNotFoundError('could not find validation file')
    
    # get the tensors out from both of them.
    train_data = ValidationLoader(train3d_tensors_path)
    val_data = ValidationLoader(val3d_tensors_path)

    compute_nearest_neighbours(train_data, val_data, args.save_path)

if __name__ == '__main__':
    main()