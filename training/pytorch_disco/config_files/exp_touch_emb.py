from exp_base import *

############## choose an experiment ##############

current = 'trainer'
current = 'builder'
current = 'train_only_touch_tensor_one_obj'
current = 'train_only_touch_two_obj'
current = 'train_forward_touch'
current = 'eval_forward_touch'
# current = 'train_moco_forward'

# # (No MODS HERE) ??
# mod = '"a00"'  # checking the loading of pretrained model
# mod = '"a01"'  # training with all objects I have curr_full_train.txt
# mod = '"a02"'  # checking why two rgbs are appearing tensorboard
# mod = '"a03"'  # fixed the above problem training now
# mod = '"a04"'  # fixing the problem which I do not yet know
# mod = '"a05"'  # I resized depths but did not rescale the intrinsics its fixed now
# mod = '"a06"'  # experiment which runs only l2 loss on embeddings see param emb_touch_do_ml
# mod = '"a07"'  # clipping depth in the range of 0 to 1 and then running the exp with only l2 loss
# mod = '"a08"'  # code with Adam's suggested changes of using bilinear_samp3d, also adding eval_recall
# mod = '"a09"'  # code with eval recall added, TODO: check with Adam if it is fine
# mod = '"a01"'  # code for testing the occs of sensor and visual input
# mod = '"a02"'  # still the same check
# mod = '"a03"'  # reverting back to ax2
# mod = '"a04"'  # fixed unps as well
# mod = '"a05"'  # sensor occRs fixing
# mod = '"a06"'  # added the valid img for the sensor
# mod = '"a07"'  # with only subsampled sensor images

# I just want to test some things
# mod = '"a00"'  # for checking what is the dimensionality of the things in margin loss

############## define experiments ##############

exps['trainer'] = ['touch_embed', # mode, this makes do_embed_touch=True
                   'touch_data', # dataset, this is defined below
                   '20k_iters',  # how many iterations I will optimizer for
                   'lr3',  # makes lr = 1e-3
                   'B8',  # makes batch_size = 8
                   'frozen_feat',  # this will make do_freeze_feat=True, and do_feat=True, redefined below
                   'train_touch2DML',  # defined below, holds a flag
                   'faster_logging',  # logging frequency is defined here
]

exps['builder'] = ['touch_embed', # mode, this makes do_embed_touch=True
                   'touch_data', # dataset, this is defined below
                   '10_iters',  # how many iterations I will optimizer for
                   'lr3',  # makes lr = 1e-3
                   'B2',  # makes batch_size = 8
                   'frozen_feat',  # this will make do_freeze_feat=True, and do_feat=True, redefined below
                   'train_touch2DML',  # defined below, holds a flag
                   'fastest_logging',  # logging frequency is defined here,
]

exps['train_only_touch_tensor_one_obj'] = ['touch_embed', # mode, this makes do_embed_touch=True
                                           'one_obj_touch_data', # dataset, this is defined below
                                           '20k_iters',  # how many iterations I will optimizer for
                                           'lr3',  # makes lr = 1e-3
                                           'B1',  # makes batch_size = 8
                                           'frozen_feat',  # this will make do_freeze_feat=True, and do_feat=True, redefined below
                                           'train_touchML',  # defined below, holds a flag
                                           'faster_logging',  # logging frequency is defined here,
]

exps['train_only_touch_two_obj'] = ['touch_embed', # mode, this makes do_embed_touch=True
                                    'two_obj_touch_data', # dataset, this is defined below
                                    '10_iters',  # how many iterations I will optimizer for
                                    'lr3',  # makes lr = 1e-3
                                    'B2',  # makes batch_size = 8
                                    'frozen_feat',  # this will make do_freeze_feat=True, and do_feat=True, redefined below
                                    'train_touchML',  # defined below, holds a flag
                                    'fastest_logging',  # logging frequency is defined here,
]

exps['train_forward_touch'] = ['touch_embed',
                               'forward_prediction_data',
                               '20k_iters',
                               'lr4',
                               'B2',
                               'frozen_feat',
                               'train_forward',
                               'train_touch_ML_forward',
                               'faster_logging']

exps['eval_forward_touch'] = ['touch_embed', # makes do_touch_embed=True
                              '9_iters',
                              'forward_prediction_test',  # dataset
                              'frozen_feat',  # freeze the visual features
                              'B1',
                              'frozen_forward',  # freeze the context net
                              'eval_recall',  # do eval recall, fills up the results dict
                              #'train_touch_ML_forward', # just here for computing the loss
                              'fastest_logging', # log every step
]

exps['train_moco_forward'] = ['touch_embed',
                             'forward_prediction_data',
                             '20k_iters',
                             'lr6',
                             'B1',
                             'frozen_feat',
                             'train_forward',
                             'train_moc',
                             #'reinitialize_nets',
                             'faster_logging',
                             'eval_recall' # makes eval recall to be true
]

############## net configs ##############

groups['frozen_feat'] = ['do_feat = True',  ## this helps me add the featnet
                         'feat_dim = 32',   ## this specified the dimension of the features. Should be same as that used for visual training
                         'do_freeze_feat = True',  ## this makes the features freezed
                         'feat_init = "04_m64x64x64_p64x64_1e-3_F32_Oc_c1_s1_Ve_d32_c1_mocml_nns2048_nps1024_dl100000_md8_close_up_visual"',
                         'reset_iter = True' # makes the start iter to be 0, will start the training all over again
]

groups['frozen_forward'] = ['do_touch_feat = True',
                            'do_touch_forward = True',
                            'contextH = 0.1',
                            'contextW = 0.1',
                            'contextD = 0.1',
                            'do_freeze_touch_feat = True',
                            'do_freeze_touch_forward = True',
                            'touch_feat_init = "02_m64x64x64_1e-5_F32f_TF32_tforward_ch.1_cw.1_cd.1_mocml_nns8192_nps1024_dl8192_md4_close_up_touch_and_visualev_re1"',
                            'touch_forward_init = "02_m64x64x64_1e-5_F32f_TF32_tforward_ch.1_cw.1_cd.1_mocml_nns8192_nps1024_dl8192_md4_close_up_touch_and_visualev_re1"',
                            'do_bn = False'
]

# this means I want to minimize the metric learning loss more heavily as compared to the l2 loss
# Now in the train touch ML occupancy prediction is also added
groups['train_touchML'] = ['do_touch_feat = True',
                           'do_touch_embML = True',
                        #    'do_touch_occ = True', look ma! not training for occs
                        #    'occ_coeff = 1.0',
                        #    'occ_do_cheap = True',
                        #    'occ_smooth_coeff = 1.0',
                           'emb_3D_num_samples = 128',
                           'emb_3D_ml_coeff = 1.0',
                           'emb_3D_l2_coeff = 0.1',
                           'emb_3D_smooth_coeff = 0.01',
                           'emb_3D_mindist = 16.0'
]

# the units of context are 0.1m, it is like in front of this camera I will look at 0.1m cube scene
groups['train_forward'] = ['do_touch_forward = True',
                           'do_touch_feat = True',
                           'contextH = 0.1',
                           'contextW = 0.1',
                           'contextD = 0.1'
]

groups['train_touch_ML_forward'] = ['do_touch_embML = True',
                                    'emb_3D_num_samples = 512',
                                    'emb_3D_ml_coeff = 1.0',
                                    'emb_3D_l2_coeff = 0.01'
]

groups['train_moc'] = ['do_moc = True',
                       'dict_len = 1024',
                       'num_neg_samples = 1024',
                       'num_pos_samples = 1024',
                       'do_bn = False',
                       'emb_3D_mindist = 4.0'
]

groups['reinitialize_nets'] = ['touch_feat_init = "/home/gauravp/pytorch_disco/checkpoints/02_m64x64x64_1e-5_F32f_TF32_tforward_ch.1_cw.1_cd.1_mocml_nns4096_nps1024_dl4096_md4_close_up_touch_and_visual_ev_re1_two_dicts"',
                               'touch_forward_init = "/home/gauravp/pytorch_disco/checkpoints/02_m64x64x64_1e-5_F32f_TF32_tforward_ch.1_cw.1_cd.1_mocml_nns4096_nps1024_dl4096_md4_close_up_touch_and_visual_ev_re1_two_dicts"']

############## datasets ##############

SIZE = 32
Z = SIZE*2 # 64
Y = SIZE*2 # 64
X = SIZE*2 # 64

H = 128
W = 128

# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

V = 16384 # pointcloud size
S = 5
sensor_S = 1024  # right now working with 1024 touches

groups['touch_data'] = ['dataset_name = "touch_data"',
                        'H = %d' % H,
                        'W = %d' % W,
                        'trainset = "curr_full_train"',
                        'dataset_location = "/home/gauravp/depth_cam_env/dataset"',
                        'dataset_list_dir = "/home/gauravp/depth_cam_env/dataset"',
                        'dataset_format = "txt"'
]

groups['one_obj_touch_data'] = ['dataset_name = "touch_data"',
                                'H = %d' % H,
                                'W = %d' % W,
                                'trainset = "one_object"',
                                'dataset_location = "/home/gauravp/depth_cam_env/dataset"',
                                'dataset_list_dir = "/home/gauravp/depth_cam_env/dataset"',
                                'dataset_format = "txt"'
]

groups['two_obj_touch_data'] = ['dataset_name = "touch_data"',
                                'H = %d' % H,
                                'W = %d' % W,
                                'trainset = "two_obj_touch_and_visual"',
                                'dataset_location = "/home/gauravp/train_files"',
                                'dataset_list_dir = "/home/gauravp/train_files"',
                                'dataset_format = "txt"'
]

## forward prediction data is same as the two object touch data, this is here for separation
groups['forward_prediction_data'] = ['dataset_name = "touch_data"',
                                     'H = %d' % H,
                                     'W = %d' % W,
                                     'trainset = "one_obj_touch_and_visual"',
                                    #  'valset = "close_up_touch_and_visual"',
                                     'dataset_location = "/home/gauravp/train_files"',
                                     'dataset_list_dir = "/home/gauravp/train_files"',
                                     'dataset_format = "txt"'
]

groups['forward_prediction_test'] = ['dataset_name = "touch_data"',
                                     'H = %d' % H,
                                     'W = %d' % W,
                                     'valset = "close_up_touch_and_visual"',
                                     'dataset_location = "/home/gauravp/train_files"',
                                     'dataset_list_dir = "/home/gauravp/train_files"',
                                     'dataset_format = "txt"'
]

############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()['s']
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
