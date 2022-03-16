import os

B = 2 # batch size
MB = 1 # batch size for metric learning

max_clusters = 2
commitment_cost = 0.25
is_refine_net = False
is_init_cluter_with_instance = False
top_grasp_only = False
H = 240 # height
W = 320 # width

# BY = 200*2 # bird height (y axis, [-40, 40])
# BX = 176*2 # bird width (x axis, [0, 70.4])
# BZ = 20 # bird depth (z axis, [-3.0, 1.0])

# MH = 200*2
# MW = 176*2
# MD = 20

Z = 128
Y = 64
X = 128

PH = int(128/4)
PW = int(384/4)


fix_crop = False
# ZY = 32
# ZX = 32
# ZZ = 16

N = 50 # number of boxes produced by the rcnn (not all are good)
K = 1 # number of boxes to actually use
S = 2 # seq length
T = 256 # height & width of birdview map
V = 100000 # num velodyne points
sensor_S = 10 # sensor length for sequence

#----------- loading -----------#
loadname = None
emb2D_init = ""
feat_init = ""
obj_init = ""
box_init = ""
ort_init = ""
inp_init = ""
traj_init = ""
occ_init = ""
view_init = ""
vis_init = ""
flow_init = ""
ego_init = ""
total_init = True
touch_feat_init = ""  # path to initialize the touch featnet
touch_forward_init = ""  # path to initialize the context net
reset_iter = False

#--------- training mode ----------#
do_compute_cluster_center = False

do_freeze_emb2D = False
do_freeze_feat = False
do_freeze_obj = False
do_freeze_box = False
do_freeze_ort = False
do_freeze_inp = False
do_freeze_traj = False
do_freeze_occ = False
do_freeze_view = False
do_freeze_vis = False
do_freeze_flow = False
do_freeze_ego = False
do_freeze_touch_feat = False
do_freeze_touch_forward = False
do_resume = False
do_profile = False

# by default, only backprop on "train" iters
backprop_on_train = True
backprop_on_val = False
backprop_on_test = False

# eval mode: save npys
do_eval_map = False
do_eval_recall = False # keep a buffer and eval recall within it
do_save_embs = False
do_save_ego = False

#----------- augs -----------#
# do_aug2D = False
# do_aug3D = False
do_aug_color = False
do_time_flip = False
do_horz_flip = False
do_synth_rt = False
do_synth_nomotion = False
do_piecewise_rt = False
do_sparsify_pointcloud = 0 # choose a number here, for # pts to use

#----------- net design -----------#
# run nothing
do_emb2D = False
do_emb3D = False
do_feat = False
do_obj = False
do_box = False
do_ort = False
do_inp = False
do_traj = False
do_occ = False
do_view = False
do_flow = False
do_ego = False
do_vis = False
do_touch_embML = False
do_touch_feat = False
do_touch_occ = False
do_touch_forward = False
do_moc = False
do_metric_learning = False
do_validation = False
do_generate_data = False
do_det = False
deeper_det = False
#----------- general hypers -----------#
lr = 0.0

#----------- emb hypers -----------#
emb_2D_smooth_coeff = 0.0
emb_3D_smooth_coeff = 0.0
emb_2D_ml_coeff = 0.0
emb_3D_ml_coeff = 0.0
emb_2D_l2_coeff = 0.0
emb_3D_l2_coeff = 0.0
emb_2D_mindist = 0.0
emb_3D_mindist = 0.0
emb_2D_num_samples = 0
emb_3D_num_samples = 0

# ..... Added for touch embedding .... #
emb_3D_touch_num_samples = 0
emb_3D_touch_mindist = 0.0
emb_3D_touch_ml_coeff = 0.0
emb_3D_touch_l2_coeff = 0.0

#----------- feat hypers -----------#
feat_coeff = 0.0
feat_rigid_coeff = 0.0
feat_do_vae = False
feat_do_sb = False
feat_do_resnet = False
feat_do_sparse_invar = False
feat_kl_coeff = 0.0
feat_dim = 8
feat_do_flip = False
feat_do_rt = False

#----------- obj hypers -----------#
obj_coeff = 0.0
obj_dim = 8

#----------- box hypers -----------#
box_sup_coeff = 0.0
box_cs_coeff = 0.0
box_dim = 8

#----------- ort hypers -----------#
ort_coeff = 0.0
ort_warp_coeff = 0.0
ort_dim = 8

#----------- inp hypers -----------#
inp_coeff = 0.0
inp_dim = 8

#----------- traj hypers -----------#
traj_coeff = 0.0
traj_dim = 8

#----------- occ hypers -----------#
occ_do_cheap = False
occ_coeff = 0.0
occ_smooth_coeff = 0.0

#----------- view hypers -----------#
view_depth = 64
view_pred_embs = False
view_pred_rgb = False
view_l1_coeff = 0.0
view_ce_coeff = 0.0
view_dl_coeff = 0.0

#----------- vis hypers-------------#
vis_softmax_coeff = 0.0
vis_hard_coeff = 0.0
vis_l1_coeff = 0.0
vis_debug = False

#----------- flow hypers -----------#
flow_warp_coeff = 0.0
flow_cycle_coeff = 0.0
flow_smooth_coeff = 0.0
flow_l1_coeff = 0.0
flow_synth_l1_coeff = 0.0
flow_do_synth_rt = False
flow_patch_size = 4

#----------- ego hypers -----------#
ego_use_gt = False
ego_use_precomputed = False
ego_rtd_coeff = 0.0
ego_rta_coeff = 0.0
ego_traj_coeff = 0.0
ego_warp_coeff = 0.0

# ---------- Place holder for forward prediction hyper if any ----------- #
contextH = 4
contextW = 4
contextD = 4

# ---- metric learning loss ---- #
metric_learning_loss_type = "cluster_id" # success_rate

# --------- moc hypers ------------- #
dict_len = 10000
num_neg_samples = 2000
do_bn = True  # Do I have the capability of doing batch normalization
num_pos_samples = 1024  # helpful for doing voxel level moco_learning

# --------- det hypers ------------- #
det_anchor_size = 12.0
det_prob_coeff = 1.0
det_reg_coeff = 1.0
alpha_pos = 1.5
beta_neg = 1.0

det_anchor_size_x = 0
det_anchor_size_y = 0
det_anchor_size_z = 0

#----------- mod -----------#

mod = '""'

############ slower-to-change hyperparams below here ############

## logging
log_freq_train = 100
log_freq_val = 100
log_freq_test = 100
snap_freq = 5000

max_iters = 10000
shuffle_train = True
shuffle_val = True
shuffle_test = True

dataset_name = ""
seqname = ""

trainset = ""
valset = ""
testset = ""

dataset_list_dir = ""
dataset_location = ""
validation_path = ""
validate_after = 1

dataset_format = "py" #can be py or npz

# mode selection
do_zoom = False
do_carla_det = False
do_carla_mot = False
do_carla_flo = False
do_carla_sta = False
do_mujoco_offline = False
do_mujoco_offline_metric = False
do_touch_embed = False

############ rev up the experiment ############


train_mode = "train"

mode = os.environ["MODE"]
print('os.environ mode is %s' % mode)
if mode=="CARLA_DET":
    exec(compile(open('config_files/exp_carla_det.py').read(), 'exp_carla_det.py', 'exec'))
elif mode=="CARLA_MOT":
    exec(compile(open('config_files/exp_carla_mot.py').read(), 'exp_carla_mot.py', 'exec'))
elif mode=="CARLA_FLO":
    exec(compile(open('config_files/exp_carla_flo.py').read(), 'exp_carla_flo.py', 'exec'))
elif mode=="CARLA_STA":
    exec(compile(open('config_files/exp_carla_sta.py').read(), 'exp_carla_sta.py', 'exec'))
elif mode=="MUJOCO_OFFLINE":
    exec(open('config_files/exp_mujoco_offline.py').read())
elif mode=="MUJOCO_OFFLINE_METRIC":
    exec(open('config_files/exp_mujoco_offline_metric.py').read())
elif mode=="MUJOCO_OFFLINE_METRIC_2D":
    exec(open('config_files/exp_mujoco_offline_metric_2d.py').read())
elif mode == "TOUCH_EMB":
    exec(compile(open('config_files/exp_touch_emb.py').read(), 'exp_touch_emb.py', 'exec'))
elif mode=="CUSTOM":
    exec(compile(open('exp_custom.py').read(), 'exp_custom.py', 'exec'))
else:
    assert(False) # what mode is this?

############ make some final adjustments ############

trainset_path = "%s/%s.txt" % (dataset_list_dir, trainset)
valset_path = "%s/%s.txt" % (dataset_list_dir, valset)
testset_path = "%s/%s.txt" % (dataset_list_dir, testset)

data_paths = {}
data_paths['train'] = trainset_path
data_paths['val'] = valset_path
data_paths['test'] = testset_path

set_nums = {}
set_nums['train'] = 0
set_nums['val'] = 1
set_nums['test'] = 2

set_names = ['train', 'val', 'test']

log_freqs = {}
log_freqs['train'] = log_freq_train
log_freqs['val'] = log_freq_val
log_freqs['test'] = log_freq_test

shuffles = {}
shuffles['train'] = shuffle_train
shuffles['val'] = shuffle_val
shuffles['test'] = shuffle_test


############ autogen a name; don't touch any hypers! ############

def strnum(x):
    s = '%g' % x
    if '.' in s:
        s = s[s.index('.'):]
    return s

name = "%02d_m%dx%dx%d" % (B, Z,Y,X)
if do_view or do_emb2D:
    name += "_p%dx%d" % (PH,PW)

if lr > 0.0:
    lrn = "%.1e" % lr
    # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]
    name += "_%s" % lrn

if do_feat:
    name += "_F"
    name += "%d" % feat_dim
    if feat_do_flip:
        name += "l"
    if feat_do_rt:
        name += "r"
    if feat_do_vae:
        name += "v"
    if feat_do_sb:
        name += 'b'
    if feat_do_resnet:
        name += 'r'
    if feat_do_sparse_invar:
        name += 'i'
    if do_freeze_feat:
        name += "f"
    else:
        feat_losses = [feat_rigid_coeff,
                       feat_kl_coeff,
        ]
        feat_prefixes = ["r",
                         "k",
        ]
        for l_, l in enumerate(feat_losses):
            if l > 0:
                name += "_%s%s" % (feat_prefixes[l_],strnum(l))

if do_touch_feat:
    name += "_TF"
    name += "%d" % feat_dim


if do_ego:
    name += "_G"
    if ego_use_gt:
        name += "gt"
    elif ego_use_precomputed:
        name += "pr"
    else:
        if do_freeze_ego:
            name += "f"
        else:
            ego_losses = [ego_rtd_coeff,
                          ego_rta_coeff,
                          ego_traj_coeff,
                          ego_warp_coeff,
            ]
            ego_prefixes = ["rtd",
                            "rta",
                            "t",
                            "w",
            ]
            for l_, l in enumerate(ego_losses):
                if l > 0:
                    name += "_%s%s" % (ego_prefixes[l_],strnum(l))

if do_obj:
    name += "_J"
    # name += "%d" % obj_dim

    if do_freeze_obj:
        name += "f"
    else:
        # no real hyps here
        pass

if do_box:
    name += "_B"
    # name += "%d" % box_dim

    if do_freeze_box:
        name += "f"
    else:
        box_coeffs = [box_sup_coeff,
                      box_cs_coeff,
                      # box_smooth_coeff,
        ]
        box_prefixes = ["su",
                        "cs",
                        # "s",
        ]
        for l_, l in enumerate(box_coeffs):
            if l > 0:
                name += "_%s%s" % (box_prefixes[l_],strnum(l))


if do_ort:
    name += "_O"
    # name += "%d" % ort_dim

    if do_freeze_ort:
        name += "f"
    else:
        ort_coeffs = [ort_coeff,
                      ort_warp_coeff,
                      # ort_smooth_coeff,
        ]
        ort_prefixes = ["c",
                        "w",
                        # "s",
        ]
        for l_, l in enumerate(ort_coeffs):
            if l > 0:
                name += "_%s%s" % (ort_prefixes[l_],strnum(l))

if do_inp:
    name += "_I"
    # name += "%d" % inp_dim

    if do_freeze_inp:
        name += "f"
    else:
        inp_coeffs = [inp_coeff,
                      # inp_smooth_coeff,
        ]
        inp_prefixes = ["c",
                        # "s",
        ]
        for l_, l in enumerate(inp_coeffs):
            if l > 0:
                name += "_%s%s" % (inp_prefixes[l_],strnum(l))

if do_traj:
    name += "_T"
    name += "%d" % traj_dim

    if do_freeze_traj:
        name += "f"
    else:
        # no real hyps here
        pass

if do_occ:
    name += "_O"
    if occ_do_cheap:
        name += "c"
    if do_freeze_occ:
        name += "f"
    else:
        occ_coeffs = [occ_coeff,
                      occ_smooth_coeff,
        ]
        occ_prefixes = ["c",
                        "s",
        ]
        for l_, l in enumerate(occ_coeffs):
            if l > 0:
                name += "_%s%s" % (occ_prefixes[l_],strnum(l))

if do_touch_occ:
    name += "_TO"
    if occ_do_cheap:
        name += "c"
    if do_freeze_occ:
        name += "f"
    else:
        occ_coeffs = [occ_coeff,
                      occ_smooth_coeff,
        ]
        occ_prefixes = ["c",
                        "s",
        ]
        for l_, l in enumerate(occ_coeffs):
            if l > 0:
                name += "_%s%s" % (occ_prefixes[l_],strnum(l))

if do_view:
    name += "_V"
    if view_pred_embs:
        name += "e"
    if view_pred_rgb:
        name += "r"
    if do_freeze_view:
        name += "f"

    # sometimes, even if view is frozen, we use the loss
    # to train other nets
    view_coeffs = [view_depth,
                   view_l1_coeff,
                   view_ce_coeff,
                   view_dl_coeff,
    ]
    view_prefixes = ["d",
                     "c",
                     "e",
                     "s",
    ]
    for l_, l in enumerate(view_coeffs):
        if l > 0:
            name += "_%s%s" % (view_prefixes[l_],strnum(l))

if do_vis:
    name += "_V"
    if vis_debug:
        name += 'd'
    if do_freeze_vis:
        name += "f"
    else:
        vis_coeffs = [vis_softmax_coeff,
                      vis_hard_coeff,
                      vis_l1_coeff,
        ]
        vis_prefixes = ["s",
                        "h",
                        "c",
        ]
        for l_, l in enumerate(vis_coeffs):
            if l > 0:
                name += "_%s%s" % (vis_prefixes[l_],strnum(l))


if do_emb2D:
    name += "_E2"
    if do_freeze_emb2D:
        name += "f"
    emb_coeffs = [emb_2D_smooth_coeff,
                  emb_2D_ml_coeff,
                  emb_2D_l2_coeff,
                  emb_2D_num_samples,
                  emb_2D_mindist,
    ]
    emb_prefixes = ["s",
                    "m",
                    "e",
                    "n",
                    "d",
    ]
    for l_, l in enumerate(emb_coeffs):
        if l > 0:
            name += "_%s%s" % (emb_prefixes[l_],strnum(l))
if do_emb3D:
    name += "_E3"
    emb_coeffs = [emb_3D_smooth_coeff,
                  emb_3D_ml_coeff,
                  emb_3D_l2_coeff,
                  emb_3D_num_samples,
                  emb_3D_mindist,
    ]
    emb_prefixes = ["s",
                    "m",
                    "e",
                    "n",
                    "d",
    ]
    for l_, l in enumerate(emb_coeffs):
        if l > 0:
            name += "_%s%s" % (emb_prefixes[l_],strnum(l))

if do_touch_embML:
    name += "_touchE3"
    emb_coeffs = [emb_3D_smooth_coeff,
                  emb_3D_ml_coeff,
                  emb_3D_l2_coeff,
                  emb_3D_num_samples,
                  emb_3D_mindist,
    ]
    emb_prefixes = ["s",
                    "m",
                    "e",
                    "n",
                    "d",
    ]
    for l_, l in enumerate(emb_coeffs):
        if l > 0:
            name += "_%s%s" % (emb_prefixes[l_],strnum(l))


if do_touch_forward:
    name += "_tforward"
    # hyperparams if any go here
    forward_vars = [contextH,
                    contextW,
                    contextD]
    forward_prefixes = ['ch', 'cw', 'cd']
    for l_, l in enumerate(forward_vars):
        if l > 0:
            name += "_%s%s" % (forward_prefixes[l_], strnum(l))

if do_moc:
    name += "_mocml"
    moc_vars = [num_neg_samples,
                num_pos_samples,
                dict_len,
                do_bn,
                emb_3D_mindist]

    moc_prefixes = ['nns', 'nps', 'dl', 'do_bn', 'md']
    for l_, l in enumerate(moc_vars):
        if l > 0:
            name += "_%s%s" % (moc_prefixes[l_], strnum(l))

if do_flow:
    name += "_F"
    if do_freeze_flow:
        name += "f"
    else:
        flow_coeffs = [flow_warp_coeff,
                       flow_cycle_coeff,
                       flow_smooth_coeff,
                       flow_l1_coeff,
                       flow_synth_l1_coeff,
        ]
        flow_prefixes = ["w",
                         "c",
                         "s",
                         "e",
                         "y",
        ]
        for l_, l in enumerate(flow_coeffs):
            if l > 0:
                name += "_%s%s" % (flow_prefixes[l_],strnum(l))

##### end model description

# add some training data info
sets_to_run = {}
if trainset:
    name = "%s_%s" % (name, trainset)
    sets_to_run['train'] = True
else:
    sets_to_run['train'] = False

if valset:
    name = "%s_%s" % (name, valset)
    sets_to_run['val'] = True
else:
    sets_to_run['val'] = False

if testset:
    name = "%s_%s" % (name, testset)
    sets_to_run['test'] = True
else:
    sets_to_run['test'] = False

sets_to_backprop = {}
sets_to_backprop['train'] = backprop_on_train
sets_to_backprop['val'] = backprop_on_val
sets_to_backprop['test'] = backprop_on_test


if (do_aug_color or
    do_horz_flip or
    do_time_flip or
    do_synth_rt or
    do_piecewise_rt or
    do_synth_nomotion or
    do_sparsify_pointcloud):
    name += "_A"
    if do_aug_color:
        name += "c"
    if do_horz_flip:
        name += "h"
    if do_time_flip:
        name += "t"
    if do_synth_rt:
        assert(not do_piecewise_rt)
        name += "s"
    if do_piecewise_rt:
        assert(not do_synth_rt)
        name += "p"
    if do_synth_nomotion:
        name += "n"
    if do_sparsify_pointcloud:
        name += "v"

if (not shuffle_train) or (not shuffle_val) or (not shuffle_test):
    name += "_ns"


if do_profile:
    name += "_PR"

if mod:
    name = "%s_%s" % (name, mod)

if do_resume:
    total_init = name

if do_eval_recall:
    name += '_ev_re1_evaluation'

if do_validation:
    splits = validation_path.split('/')
    val_path = splits[-1][:-4]
    name += f'val_{val_path}'

print(name)
