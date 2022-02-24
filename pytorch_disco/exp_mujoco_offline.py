from config_files.exp_base import *

############## choose an experiment ##############

current = os.environ["exp_name"]

#current = 'builder'
#current = 'trainer'
#current = 'train_visual_tensor'
#current = 'train_visual_tensor_moc'
#current = 'train_viewpred_occ'
#current = 'train_quant_vis_viewpred_occ_dt1542020'

#current = 'quant_generate_data'

# current = 'train_metric_learner'
# current = 'eval_quant_vis'
# current = 'train_only_moco'

# (No MODS HERE) ??
# mod = '"a00"' # changed the range to discretize in utils.py
# mod = "'a01'"  # one with the train and validation data separate
# mod = "'a02'"  # one with bilinear mode sampling debug
# mod = "'a03'"  # for training with possibly fixed depth issue
# mod = "'a04'"  # putting the wrong center to make sure I get something weird
# mod = "'a05'"  # first attempt to train with multiple objects at once
# mod = "'a06'"  # fixing the occs issue of visualization
# mod = "'a07'"  # issue with occs fixed by changing the view to frontal
# mod = "'a08'"  # checking what is the issue with inputs in touch by comparing against this
# mod = "'a09'"  # did I introduce some bug
# mod = "'a10'"  # checking with the new data if everything is fine
# mod = "'a01'"  # just want to check if using the right bounds
# mod = "'a01'"  # I am doing some checks to see if I have made a mistake in moco.
# mod = "'a02'"  # Reducing the learning rate to prevent jumping around 3e-4
# mod = "'a03'"  # Reducing the learning rate to prevent jumping around 1e-4

############## define experiments ##############

exps['builder'] = ['mujoco_offline', # mode
                   'mujoco_offline_data', # dataset
                   '10_iters',
                   'lr0',
                   'B1',
                   'train_feat',
                   'fastest_logging',
]
exps['trainer'] = ['mujoco_offline', # mode
                   'mujoco_offline_data', # dataset
                   '40k_iters',
                   'lr3',
                   'B1',
                   'train_feat',
                   'train_occ',
                   'train_emb_view',
                   'faster_logging',
]

exps['train_visual_tensor'] = ['mujoco_offline', # mode
                               'mujoco_offline_data', # dataset
                               '40k_iters',
                               'lr3',
                               'B4',
                               'train_feat',
                               'train_occ',
                               'train_emb_view',
                               'faster_logging',
]

exps['train_visual_tensor_moc'] = ['mujoco_offline',
                                   'mujoco_offline_data',
                                   '40k_iters',
                                   'lr3',
                                   'B4',
                                   'train_feat',  # do_feat=True, feat_dim=32
                                   'train_occ',   # occnet
                                   'train_view',  # view prediction
                                   'train_moc',   # moco_net
                                   'faster_logging'
]

exps['train_only_moco'] = ['mujoco_offline',
                          'mujoco_offline_data',
                          '40k_iters',
                          'lr4',
                          'B4',
                          'train_feat',
                          'train_moc',
                          'faster_logging']

exps['train_viewpred_occ'] = ['mujoco_offline',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_data',
                              '40k_iters', # num_iters
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']


exps['train_quant_vis_viewpred_occ_dt1542020'] = ['mujoco_offline',  # mode
                                        'mujoco_offline_data_dt1542020',  # input format
                                        '40k_iters', # num_iters
                                        'lr3', # 1e-3
                                        'B2',  # batch_size
                                        'train_feat',  # train the feats
                                        'train_occ',   # train the occs
                                        'train_view',  # train the view
                                        'validate',    # do the nearest neighbor vis
                                        'faster_logging']

exps['quant_generate_data'] = ['mujoco_offline',
                               'mujoco_offline_data_generate_data',
                               '100_iters',  # because there are only 10 files in the valset
                               'B1',   # just want to make life easier
                               'frozen_feat',  # freeze the featnet
                               'frozen_occ',   # freeze the occnet
                               'frozen_view',  # freeze the viewnet
                               'generate_data',  # do do_generate_data = True
                               'fastest_logging']

exps['train_metric_learner'] = ['mujoco_offline',
                                'mujoco_offline_data',
                                '40k_iters',
                                'B1', #Do not tune this. Should always be 1
                                'MB2', #Increase number of elements in the batch with this parameter
                                'lr3',
                                'frozen_feat',  # do_feat=True, feat_dim=32
                                'frozen_occ',   # occnet
                                'frozen_view',  # view prediction
                                'eval_recall',
                                'metric_learning',
                                'fastest_logging']

############## net configs ##############

# featnet
groups['train_feat'] = ['do_feat = True',
                        'feat_dim = 32',
]
# occnet
groups['train_occ'] = ['do_occ = True',
                       'occ_coeff = 1.0',
                       'occ_do_cheap = True',
                       'occ_smooth_coeff = 1.0',
]

# viewnet
groups['train_view'] = ['do_view = True',
                        'view_depth = 32',
                        'view_pred_embs = True',
                        'view_l1_coeff = 1.0',

]

groups['frozen_feat'] = ['do_feat = True',
                         'do_freeze_feat = True',
                         'feat_dim = 32']

groups['frozen_occ'] = ['do_occ = True',
                        'do_freeze_occ = True',
                        'occ_coeff = 1.0',
                        'occ_do_cheap = True',
                        'occ_smooth_coeff = 1.0']

groups['frozen_view'] = ['do_view = True',
                         'do_freeze_view = True',
                         'view_depth = 32',
                         'view_pred_embs = True',
                         'view_l1_coeff = 1.0']

groups['generate_data'] = ['do_generate_data = True',
                           'feat_init = "02_m64x64x64_p64x64_1e-3_F32_Oc_c1_s1_Ve_d32_c1_train_file_controller_dt1542020val_train_val_file_dt1542020"',
                           'occ_init = "02_m64x64x64_p64x64_1e-3_F32_Oc_c1_s1_Ve_d32_c1_train_file_controller_dt1542020val_train_val_file_dt1542020"',
                           'view_init = "02_m64x64x64_p64x64_1e-3_F32_Oc_c1_s1_Ve_d32_c1_train_file_controller_dt1542020val_train_val_file_dt1542020"',
                           'reset_iter = True'
                          ]

groups['metric_learning'] = ['do_metric_learning = True']

# combination of view-net, emb-net and emb-net3D
groups['train_emb_view'] = [
    'do_view = True',
    'do_emb2D = True',
    'do_emb3D = True',
    'view_depth = 32',
    'emb_2D_ml_coeff = 1.0',
    'emb_3D_ml_coeff = 1.0',
    'emb_samp = "rand"',
    'emb_dim = 32',
    'view_pred_embs = True',
    'do_eval_recall = True',
    'emb_2D_smooth_coeff = 0.01',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2'
]

# moco-net
groups['train_moc'] = ['do_moc = True',
                       'dict_len = 100000',
                       'num_neg_samples = 2048',
                       'do_bn = False',
                       'num_pos_samples = 1024',
                       'emb_3D_mindist = 8.0']

groups['validate'] = ['do_validation = True',
                      'validate_after = 50']

groups['validate_dt1542020.txt'] = ['do_validation = True',
                      'validation_path = "backend/train_files/train_val_file_dt1542020.txt"',
                      'validate_after = 100']

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
S = 3

# validation_path = "/home/ubuntu/pytorch_disco/backend/inputs/train_files/train_val_files.txt"

groups['mujoco_offline_data'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "train_file_controller"',
                                 'valset = "val_file_controller"',
                                 'validation_path = "train_val_files.txt"',
                                 'dataset_list_dir = "backend/train_files"',
                                 'dataset_format = "txt"'
]


groups['grasptop_all_data'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "parsed_task_grasptop_all_train"',
                                 'trainvalset = "parsed_task_grasptop_all_train_val"',
                                 'valset = "parsed_task_grasptop_all_val"',
                                 'dataset_list_dir = "/projects/katefgroup/quantized_policies/data/grasptop"',
                                 'dataset_format = "txt"'
]



groups['mujoco_offline_data_dt1542020'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "train_file_controller_dt1542020"',
                                 'valset = "val_file_controller_dt1542020"',
                                 'dataset_list_dir = "backend/train_files"',
                                 'dataset_format = "txt"'
]

#groups['mujoco_offline_data_generate_data'] = ['dataset_name = "mujoco_offline"',
#                                               'H = %d' %H,
#                                               'W = %d' %W,
#                                               'valset = "train_file_controller_dt1542020"',
#                                               'testset = "val_file_controller_dt1542020"',
#                                               'dataset_list_dir = "/home/ubuntu/pytorch_disco/backend/train_files"',
#                                               'dataset_format = "txt"']

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
