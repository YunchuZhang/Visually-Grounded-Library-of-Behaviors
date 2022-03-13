from exp_base import *

# THIS FILE IS FOR STORING STANDARD EXPERIMENTS/BASELINES FOR CARLA_STA MODE

############## choose an experiment ##############

# current = 'builder'
# current = 'trainer_sb'
# current = 'builder'
current = 'res_trainer'
# current = 'vis_trainer'
# current = 'occvis_trainer'
# current = 'emb_trainer_sb'
# current = 'emb_trainer'
# current = 'emb_trainer_kitti'
# current = 'tow_trainer'

# (NO MODS HERE)
mod = '""'


############## define experiments ##############

exps['builder'] = ['carla_sta', # mode
                   'carla_sta10_data', # dataset
                   '3_iters',
                   'lr0',
                   'B2',
                   'no_shuf',
                   # 'no_backprop',
                   'train_feat',
                   'train_occ',
                   'train_view',
                   'train_emb',
                   'fastest_logging',
]

exps['trainer'] = ['carla_sta', # mode
                   'carla_sta_data', # dataset
                   '200k_iters',
                   'lr3',
                   'B4',
                   'train_feat',
                   'train_occ',
                   'train_view',
                   'train_emb',
                   # 'pretrained_carl_feat',
                   # 'pretrained_carl_occ',
                   # 'pretrained_carl_view',
                   # 'pretrained_carl_emb',
                   'faster_logging',
                   # 'resume'
]

exps['res_trainer'] = ['carla_sta', # mode
                   'carla_sta_data', # dataset
                   '200k_iters',
                   'lr3',
                   'B4',
                   'train_feat_res',
                   'train_occ',
                   'train_view',
                   'train_emb',
                   # 'pretrained_carl_feat',
                   # 'pretrained_carl_occ',
                   # 'pretrained_carl_view',
                   # 'pretrained_carl_emb',
                   'faster_logging',
                   # 'resume'
]

exps['emb_trainer'] = ['carla_sta', # mode
                       'carla_static_data', # dataset
                       '300k_iters',
                       'lr3',
                       'B1',
                       'train_feat',
                       'train_occ',
                       'train_emb_view',
                       'faster_logging',
]
exps['trainer_sb'] = ['carla_sta', # mode
                       'carla_sta_data', # dataset
                       '300k_iters',
                       'lr3',
                       'B4',
                       'train_feat_sb',
                       'train_occ_notcheap',
                       'train_view',
                       'train_emb',
                       'faster_logging',
                       #'fast_logging',
                       #'fastest_logging',
]
exps['emb_trainer_noocc'] = ['carla_sta', # mode
                             'carla_static_data', # dataset
                             '300k_iters',
                             'lr3',
                             'B2',
                             'train_feat',
                             'train_emb_view',
                             'resume',
                             'slow_logging',
]
exps['emb_trainer_kitti'] = ['carla_sta', # mode
                             'kitti_static_data', # dataset
                             '300k_iters',
                             'lr3',
                             'B2',
                             'train_feat',
                             'train_occ',
                             'train_emb_view',
                             'fast_logging',
                             # 'synth_rt',
                             # 'resume',
                             # 'pretrained_carl_feat', 
                             # 'pretrained_carl_view', 
                             # 'pretrained_carl_emb',
                             # 'pretrained_carl_occ', 
]

exps['tow_trainer'] = ['carla_sta', # mode
                       'carla_static_data', # dataset
                       '100k_iters',
                       'lr4',
                       'B4',
                       'train_tow',
                       'fast_logging',
]

exps['vis_trainer'] = ['carla_sta', # mode
                       'carla_static_data', # dataset
                       '50k_iters',
                       'lr3',
                       'B2',
                       'pretrained_carl_occ',
                       'pretrained_carl_vis',
                       'frozen_occ',
                       'frozen_vis',
                       'train_feat',
                       'train_emb',
                       'slow_logging',
]

exps['occvis_trainer'] = ['carla_sta', # mode
                          'carla_static_data', # dataset
                          '200k_iters',
                          'lr3',
                          'B4',
                          'train_occ',
                          'train_vis',
                          'slow_logging',
]


############## net configs ##############

groups['train_box'] = ['do_box = True',
                       'box_sup_coeff = 0.01', # penalty for expanding the box min/max range
                       # 'box_cs_coeff = 1.0', # center-surround loss
]
groups['train_ort'] = ['do_ort = True',
                       # 'ort_coeff = 1.0', # sup loss (for debug)
                       'ort_warp_coeff = 1.0', # weight on 3D loss against the sta tensors
]
groups['train_inp'] = ['do_inp = True',
                       'inp_coeff = 1.0',
                       # 'inp_dim = 8', # size of bottleneck maybe; currently unused
]

groups['train_traj'] = ['do_traj = True',
                        'traj_dim = 8',
]

groups['train_feat'] = ['do_feat = True',
                        'feat_dim = 32',
                        'feat_do_rt = True',
                        'feat_do_flip = True',
                        # 'feat_dim = 16',
                        # 'feat_dim = 8',
]
groups['train_feat_res'] = ['do_feat = True',
                        'feat_dim = 32',
                        'feat_do_rt = True',
                        'feat_do_flip = True',
                        'feat_do_res = True',
                        # 'feat_dim = 16',
                        # 'feat_dim = 8',
]
groups['train_feat_sb'] = ['do_feat = True',
                        'feat_dim = 32',
                        'feat_do_sb = True',
                        'feat_do_res = True',
                        'feat_do_flip = True',
                        'feat_do_rt = True',
                        # 'feat_dim = 16',
                        # 'feat_dim = 8',
]
groups['train_feat_vae'] = ['do_feat = True',
                            'feat_dim = 32',
                            'feat_do_vae = True',
                            'feat_kl_coeff = 1.0',
]
groups['train_occ'] = ['do_occ = True',
                       'occ_do_cheap = True',
                       'occ_coeff = 1.0', 
                       'occ_smooth_coeff = 1.0', 
]
groups['train_view'] = ['do_view = True',
                       'view_depth = 32',
                       'view_l1_coeff = 1.0',
]

groups['train_occ_notcheap'] = ['do_occ = True',
                       'occ_coeff = 1.0',
                       'occ_do_cheap = False',
                       'occ_smooth_coeff = 0.1',
]
# 02_m32x128x128_p64x192_1e-3_F16_Oc_c1_s.1_Ve_d32_E16_c1_l.01_d.1_m.01_cals2c1o0t_cals2c1o0v_e17

# 02_m32x128x128_p64x192_1e-3_F32_Oc_c1_s.1_Ve_d32_E32_a1_i1_cals2c1o0t_cals2c1o0v_caos2c0o1v_i12
groups['train_emb_view'] = [
    'do_view = True',
    'do_emb = True',
    'view_depth = 32',
    'emb_2D_coeff = 1.0',
    'emb_3D_coeff = 1.0',
    'emb_samp = "rand"',
    'emb_dim = 32',
    'view_pred_embs = True',
    'do_eval_recall = True', 
]
groups['train_rgb_view'] = ['do_view = True',
                            'view_pred_rgb = True',
                            'view_use_halftanh = True',
                            # 'view_l1_coeff = 1.0', # 2d to rgb consistency
                            'view_ce_coeff = 1.0', # 2d to rgb consistency
                            'view_depth = 32',
                            'do_eval_recall = True', 
]
groups['train_tow'] = ['do_tow = True',
                       'tow_view_coeff = 1.0',
                       'tow_kl_coeff = 1.0',
                       'do_eval_recall = True', 
]
groups['train_vis'] = ['do_vis = True',
                       # 'vis_debug = True',
                       'vis_softmax_coeff = 1.0',
                       'vis_hard_coeff = 1.0',
                       'view_depth = 32',
]
groups['train_flow'] = ['do_flow = True',
                        'flow_huber_coeff = 1.0',
                        'flow_smooth_coeff = 0.01',
                        # 'flow_coeff = 10.0',
                        # 'flow_rgb_coeff = 1.0',
                        # 'flow_smooth_coeff = 40.0',
                        # 'flow_smooth_coeff = 30.0',
                        # 'flow_smooth_coeff = 20.0',
                        # 'flow_smooth_coeff = 10.0',
                        # 'flow_smooth_coeff = 5.0',
                        # 'flow_smooth_coeff = 2.0',
                        # 'snap_freq = 500', 
]
groups['train_emb'] = ['do_emb = True',
                       'emb_smooth_coeff = 0.1', 
                       'emb_2D_ml_coeff = 1.0', 
                       'emb_2D_l2_coeff = 0.1', 
                       'emb_3D_ml_coeff = 1.0', 
                       'emb_3D_l2_coeff = 0.1', 
]

############## datasets ##############

# DHW for mem stuff
SIZE = 32
Z = SIZE*4
Y = SIZE*1
X = SIZE*4

K = 2 # how many objects to consider

S = 2
H = 128
W = 384
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

groups['carla_sta1_data'] = ['dataset_name = "carla"',
                             'H = %d' % H,
                             'W = %d' % W,
                             'trainset = "caus2i6c1o0one"',
                             'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                             'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
]
groups['carla_sta10_data'] = ['dataset_name = "carla"',
                              'H = %d' % H,
                              'W = %d' % W,
                              'trainset = "caus2i6c1o0ten"',
                              'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                              'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
]
groups['carla_sta_data'] = ['dataset_name = "carla"',
                            'H = %d' % H,
                            'W = %d' % W,
                            #'trainset = "caus2i6c1o0t"',
                            'trainset = "caas2i6c0o1t"',
                            #'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_list_dir = "/projects/katefgroup/datasets/multistage_dyno/carla/npzrs"',
                            #'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/tfrs"',
                            'dataset_location = "/projects/katefgroup/datasets/multistage_dyno/carla/npzrs"',
                            'dataset_format = "npz"'
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
