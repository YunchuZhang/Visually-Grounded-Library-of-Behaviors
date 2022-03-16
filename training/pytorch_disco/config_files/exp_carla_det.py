from exp_base import *

############## choose an experiment ##############

# current = 'det_builder'
current = 'multiview_trainer'
# current = 'det_trainer'

mod = '"believe00"' # axbox
mod = '"believe01"' # axbox
mod = '"believe02"' # axbox
mod = '"believe03"' # axbox
mod = '"believe04"' # fire detnet
mod = '"believe05"' # clean up
mod = '"believe06"' # vis outs
mod = '"believe07"' # don't log so much; do backprop
mod = '"believe08"' # train
mod = '"believe09"' # more bug fixes
mod = '"believe10"' # show maps
mod = '"believe11"' # overlaps 0.33, 0.5, 0.75
mod = '"believe12"' # %.2f
mod = '"believe13"' # also use other objectives
mod = '"believe14"' # higher res
mod = '"believe15"' # 33, 50, 70
mod = '"believe16"' # play with res
mod = '"believe17"' # play with res
mod = '"believe18"' # B1, window_sz = 3
mod = '"believe19"' # B1; search for the nan
mod = '"believe20"' # same, diff gpu
mod = '"believe21"' # B2; make sure an obj exists in both frames
mod = '"believe22"' # fewer prints
mod = '"believe23"' # if any el has 0 objects, return early
mod = '"believe24"' # full data
mod = '"believe25"' # full data; 200k iters
mod = '"believe26"' # replace last layer with trilinear upsample

############## define experiments ##############

exps['det_builder'] = [
    'carla_det', # mode
    'carla_sta10_data', # dataset
    # '3_iters',
    '1k_iters',
    'train_feat',
    'train_det',
    'B1',
    'no_shuf',
    # 'no_backprop',
    'faster_logging',
]
exps['multiview_trainer'] = [
    'carla_det', # mode
    'carla_stat_stav_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_occ',
    'train_view',
    'train_emb2D',
    'train_emb3D',
    'train_det',
    'fast_logging',
]
exps['det_trainer'] = [
    'carla_det', # mode
    'carla_stat_stav_data', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_det',
    'fast_logging', 
]

############## group configs ##############

groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
    # 'feat_do_rt = True',
    # 'feat_do_flip = True',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 1.0',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_smooth_coeff = 0.01',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_det'] = [
    'do_det = True',
    'det_prob_coeff = 1.0',
    'det_reg_coeff = 1.0',
    # 'do_eval_map = True',
    'snap_freq = 5000',
]


############## datasets ##############

# DHW for mem stuff
SIZE = 32
Z = int(SIZE*4)
Y = int(SIZE*0.5)
X = int(SIZE*4)

K = 8 # how many proposals to consider

S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['carla_sta10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "caas7i6c1o0ten"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
]
groups['carla_det1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "picked"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
]
groups['carla_det_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "cabs16i3c0o1t"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
    'max_iters = 4313',
]
groups['carla_det_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "cabs16i3c0o1v"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
    'max_iters = 2124',
]
groups['carla_stat_stav_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "caas7i6c1o0t"',
    'valset = "caas7i6c1o0v"',
    'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzs"',
    'dataset_format = "npz"',
]
# DATA_MOD="as"
groups['kitti_static_data'] = ['dataset_name = "kitti"',
                               'H = %d' % H,
                               'W = %d' % W,
                               'trainset = "%st"' % (DATA_MOD),
                               'valset = "%sv"' % (DATA_MOD),
                               'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/kitti/tfrs"',
                               'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/kitti/tfrs"',
]


############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    assert group in groups
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s) 

s = "mod = " + mod
_verify_(s)

exec(s)
