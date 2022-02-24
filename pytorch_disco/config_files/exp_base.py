import pretrained_nets_carla as pret_carl

exps = {}
groups = {}

############## preprocessing/shuffling ##############

############## modes ##############


groups['zoom'] = ['do_zoom = True']
groups['carla_det'] = ['do_carla_det = True']
groups['carla_mot'] = ['do_carla_mot = True']
groups['carla_flo'] = ['do_carla_flo = True']
groups['carla_sta'] = ['do_carla_sta = True']
groups['mujoco_offline'] = ['do_mujoco_offline = True']
groups['mujoco_offline_metric'] = ['do_mujoco_offline_metric = True']
groups['mujoco_offline_metric_2d'] = ['do_mujoco_offline_metric_2d = True']
groups['touch_embed'] = ['do_touch_embed = True']


############## mujoco ##############
groups['fix_crop'] = ['fix_crop = True']

############## extras ##############

groups['decay_lr'] = ['do_decay_lr = True']
groups['clip_grad'] = ['do_clip_grad = True']
groups['quick_snap'] = ['snap_freq = 500']
groups['quicker_snap'] = ['snap_freq = 50']
groups['quickest_snap'] = ['snap_freq = 5']

groups['no_shuf'] = ['shuffle_train = False',
                     'shuffle_val = False',
                     'shuffle_test = False',
]
groups['test_mode'] = ['train_mode = "test"',
                         'backprop_on_train = False',
                         'backprop_on_val = False',
                         'backprop_on_test = False',
]
groups['gt_ego'] = ['ego_use_gt = True']
groups['precomputed_ego'] = ['ego_use_precomputed = True']
groups['aug3D'] = ['do_aug3D = True']
groups['aug2D'] = ['do_aug2D = True']

groups['sparsify_pointcloud_10k'] = ['do_sparsify_pointcloud = 10000']
groups['sparsify_pointcloud_1k'] = ['do_sparsify_pointcloud = 1000']

groups['horz_flip'] = ['do_horz_flip = True']
groups['synth_rt'] = ['do_synth_rt = True']
groups['piecewise_rt'] = ['do_piecewise_rt = True']
groups['synth_nomotion'] = ['do_synth_nomotion = True']
groups['aug_color'] = ['do_aug_color = True']
# groups['eval'] = ['do_eval = True']
groups['eval_recall'] = ['do_eval_recall = True']
groups['eval_map'] = ['do_eval_map = True']
groups['no_eval_recall'] = ['do_eval_recall = False']
groups['save_embs'] = ['do_save_embs = True']
groups['save_ego'] = ['do_save_ego = True']

groups['profile'] = ['do_profile = True',
                     'log_freq_train = 100000000',
                     'log_freq_val = 100000000',
                     'log_freq_test = 100000000',
                     'max_iters = 20']

groups['B1'] = ['B = 1']
groups['B2'] = ['B = 2']
groups['B4'] = ['B = 4']
groups['B8'] = ['B = 8']
groups['B10'] = ['B = 10']
groups['B16'] = ['B = 16']
groups['B32'] = ['B = 32']
groups['B64'] = ['B = 64']
groups['B128'] = ['B = 128']

# These are batch sizes for metric learning
groups['MB1'] = ['MB = 1']
groups['MB2'] = ['MB = 2']
groups['MB4'] = ['MB = 4']
groups['MB8'] = ['MB = 8']
groups['MB10'] = ['MB = 10']
groups['MB16'] = ['MB = 16']
groups['MB32'] = ['MB = 32']
groups['MB64'] = ['MB = 64']
groups['MB128'] = ['MB = 128']

groups['lr0'] = ['lr = 0.0']
groups['lr1'] = ['lr = 1e-1']
groups['lr2'] = ['lr = 1e-2']
groups['lr3'] = ['lr = 1e-3']
groups['2lr4'] = ['lr = 2e-4']
groups['5lr4'] = ['lr = 5e-4']
groups['lr4'] = ['lr = 1e-4']
groups['lr5'] = ['lr = 1e-5']
groups['lr6'] = ['lr = 1e-6']
groups['lr7'] = ['lr = 1e-7']
groups['lr8'] = ['lr = 1e-8']
groups['lr9'] = ['lr = 1e-9']
groups['lr12'] = ['lr = 1e-12']
groups['1_iters'] = ['max_iters = 1']
groups['2_iters'] = ['max_iters = 2']
groups['3_iters'] = ['max_iters = 3']
groups['5_iters'] = ['max_iters = 5']
groups['6_iters'] = ['max_iters = 6']
groups['9_iters'] = ['max_iters = 9']
groups['21_iters'] = ['max_iters = 21']
groups['10_iters'] = ['max_iters = 10']
groups['20_iters'] = ['max_iters = 20']
groups['25_iters'] = ['max_iters = 25']
groups['30_iters'] = ['max_iters = 30']
groups['50_iters'] = ['max_iters = 50']
groups['100_iters'] = ['max_iters = 100']
groups['150_iters'] = ['max_iters = 150']
groups['200_iters'] = ['max_iters = 200']
groups['250_iters'] = ['max_iters = 250']
groups['300_iters'] = ['max_iters = 300']
groups['397_iters'] = ['max_iters = 397']
groups['400_iters'] = ['max_iters = 400']
groups['447_iters'] = ['max_iters = 447']
groups['500_iters'] = ['max_iters = 500']
groups['850_iters'] = ['max_iters = 850']
groups['1000_iters'] = ['max_iters = 1000']
groups['2000_iters'] = ['max_iters = 2000']
groups['2445_iters'] = ['max_iters = 2445']
groups['3000_iters'] = ['max_iters = 3000']
groups['4000_iters'] = ['max_iters = 4000']
groups['4433_iters'] = ['max_iters = 4433']
groups['5000_iters'] = ['max_iters = 5000']
groups['10000_iters'] = ['max_iters = 10000']
groups['1k_iters'] = ['max_iters = 1000']
groups['2k_iters'] = ['max_iters = 2000']
groups['5k_iters'] = ['max_iters = 5000']
groups['10k_iters'] = ['max_iters = 10000']
groups['20k_iters'] = ['max_iters = 20000']
groups['30k_iters'] = ['max_iters = 30000']
groups['40k_iters'] = ['max_iters = 40000']
groups['41k_iters'] = ['max_iters = 41100']
groups['50k_iters'] = ['max_iters = 50000']
groups['60k_iters'] = ['max_iters = 60000']
groups['80k_iters'] = ['max_iters = 80000']
groups['100k_iters'] = ['max_iters = 100000']
groups['100k10_iters'] = ['max_iters = 100010']
groups['200k_iters'] = ['max_iters = 200000']
groups['300k_iters'] = ['max_iters = 300000']
groups['400k_iters'] = ['max_iters = 400000']
groups['500k_iters'] = ['max_iters = 500000']

groups['resume'] = ['do_resume = True']

groups['reset_iter'] = ['reset_iter = True']

groups['fastest_logging'] = ['log_freq_train = 1',
                             'log_freq_val = 1',
                             'log_freq_test = 1',
]
groups['faster_logging'] = ['log_freq_train = 50',
                            'log_freq_val = 50',
                            'log_freq_test = 50',
]
groups['fast_logging'] = ['log_freq_train = 250',
                          'log_freq_val = 250',
                          'log_freq_test = 250',
]
groups['slow_logging'] = ['log_freq_train = 500',
                          'log_freq_val = 500',
                          'log_freq_test = 500',
]
groups['slower_logging'] = ['log_freq_train = 1000',
                            'log_freq_val = 1000',
                            'log_freq_test = 1000',
]
groups['no_logging'] = ['log_freq_train = 100000000000',
                        'log_freq_val = 100000000000',
                        'log_freq_test = 100000000000',
]

# ############## pretrained nets ##############
groups['pretrained_carl_feat'] = ['do_feat = True',
                                  'feat_init = "' + pret_carl.feat_init + '"',
                                  # 'feat_do_vae = ' + str(pret_carl.feat_do_vae),
                                  # 'feat_dim = %d' % pret_carl.feat_dim,
]
# groups['pretrained_carl_inp'] = ['do_inp = True',
#                                  'inp_init = "' + pret_carl.inp_init + '"',
# ]
groups['pretrained_carl_view'] = ['do_view = True',
                                  'view_init = "' + pret_carl.view_init + '"',
                                  # 'view_depth = %d' %  pret_carl.view_depth,
                                  # 'view_use_halftanh = ' + str(pret_carl.view_use_halftanh),
                                  # 'view_pred_embs = ' + str(pret_carl.view_pred_embs),
                                  # 'view_pred_rgb = ' + str(pret_carl.view_pred_rgb),
]
groups['pretrained_carl_flow'] = ['do_flow = True',
                                  'flow_init = "' + pret_carl.flow_init + '"',
]
groups['pretrained_carl_tow'] = ['do_tow = True',
                                 'tow_init = "' + pret_carl.tow_init + '"',
]
groups['pretrained_carl_emb2D'] = ['do_emb = True',
                                   'emb2D_init = "' + pret_carl.emb2D_init + '"',
                                   # 'emb_dim = %d' % pret_carl.emb_dim,
]
groups['pretrained_carl_occ'] = ['do_occ = True',
                                 'occ_init = "' + pret_carl.occ_init + '"',
                                 # 'occ_do_cheap = ' + str(pret_carl.occ_do_cheap),
]
groups['pretrained_carl_vis'] = ['do_vis = True',
                                 'vis_init = "' + pret_carl.vis_init + '"',
                                 # 'occ_cheap = ' + str(pret_carl.occ_cheap),
]

groups['frozen_feat'] = ['do_freeze_feat = True', 'do_feat = True']
groups['frozen_view'] = ['do_freeze_view = True', 'do_view = True']
groups['frozen_vis'] = ['do_freeze_vis = True', 'do_vis = True']
groups['frozen_flow'] = ['do_freeze_flow = True', 'do_flow = True']
groups['frozen_emb'] = ['do_freeze_emb = True', 'do_emb = True']
groups['frozen_occ'] = ['do_freeze_occ = True', 'do_occ = True']
groups['frozen_forward'] = ['do_freeze_touch_feat = True', 'do_freeze_touch_forward = True']
# groups['frozen_ego'] = ['do_freeze_ego = True', 'do_ego = True']
# groups['frozen_inp'] = ['do_freeze_inp = True', 'do_inp = True']
groups['validate'] = [
    'do_validation = True',
    'validation_path = None',
    'validate_after = 50',
]
groups['generate_data'] = ['do_generate_data = True']