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

############## define experiments #############

exps['builder'] = ['mujoco_offline_metric', # mode
                   'mujoco_offline_data', # dataset
                   '10_iters',
                   'lr0',
                   'B1',
                   'train_feat',
                   'fastest_logging',
]
exps['trainer'] = ['mujoco_offline_metric', # mode
                   'mujoco_offline_data', # dataset
                   '40k_iters',
                   'lr3',
                   'B1',
                   'train_feat',
                   'train_occ',
                   'train_emb_view',
                   'faster_logging',
]

exps['train_visual_tensor'] = ['mujoco_offline_metric', # mode
                               'mujoco_offline_data', # dataset
                               '40k_iters',
                               'lr3',
                               'B4',
                               'train_feat',
                               'train_occ',
                               'train_emb_view',
                               'faster_logging',
]

exps['train_visual_tensor_moc'] = ['mujoco_offline_metric',
                                   'mujoco_offline_data',
                                   '40k_iters',
                                   'lr3',
                                   'B4',
                                   'train_feat',  # do_feat=True, feat_dim=32
                                   'train_occ',   # occnet
                                   'train_view',  # view prediction
                                   'train_moc',   # moco_net
                                   'slow_logging'
]

exps['train_only_moco'] = ['mujoco_offline_metric',
                          'mujoco_offline_data',
                          '40k_iters',
                          'lr4',
                          'B4',
                          'train_feat',
                          'train_moc',
                          'faster_logging']


exps['train_viewpred_occ'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_data',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']

exps['train_0629'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_0629_data',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              #'train_occ',   # train the occs
                              #'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']


exps['train_0629_refine'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_0629_data',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              #'train_occ',   # train the occs
                              #'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']


exps['train_0715_refine'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              #'train_occ',   # train the occs
                              #'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']


exps['train_0715_refine_success_rate'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '80k_iters', # num_iters
                              'is_refine_net',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              #'train_occ',   # train the occs
                              #'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']

exps['train_0715_refine_success_rate_occ_view'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']


exps['train_0715_refine_sr_occv_fixcrop_top'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_merge',
                              'top_grasp_only',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']




groups['load_train_0715_refine_sr_occv_fixcrop_top'] = [
                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/train_0715_refine_sr_occv_fixcrop_top/\'}'
                ]


exps['test_train_0715_refine_sr_occv_fixcrop_top_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_1030_franka',
                              'load_train_0715_refine_sr_occv_fixcrop_top',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '50k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'test_mode',
                              'fatest_validate',
                              'lr12', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              #'validate',    # do the nearest neighbor vis
                              'fastest_logging']



exps['test_train_0715_refine_sr_occv_fixcrop_top_franka2'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_1030_franka2',
                              'load_train_0715_refine_sr_occv_fixcrop_top',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '50k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'test_mode',
                              'fatest_validate',
                              'lr12', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              #'validate',    # do the nearest neighbor vis
                              'fastest_logging']


exps['test_train_0715_refine_sr_occv_fixcrop_top_franka3'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_1030_franka3',
                              'load_train_0715_refine_sr_occv_fixcrop_top',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '50k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'test_mode',
                              'fatest_validate',
                              'lr12', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              #'validate',    # do the nearest neighbor vis
                              'fastest_logging']


exps['finetune_train_0715_refine_sr_occv_fixcrop_top_franka3'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_1030_franka3_traineval',
                              'load_train_0715_refine_sr_occv_fixcrop_top',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '50k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              'quick_snap',
                              "metric_learning_loss_type_sr",
                              #'test_mode',
                              #'fatest_validate',
                              'lr5', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'fastest_logging']

groups['load_train_0715_st1_det_refine_sr_occ_view_fixview'] = [

                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/train_0715_st1_refine_sr_occ_view_fixview/\'}'
                ]


exps['train_0715_st1_refine_sr_occ_view_fixview'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_st1_merge',
                              'cluster_30',
                              #'load_train_0715_st1_det_refine_sr_occ_view_fixview',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'lr4', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']

exps['train_0715_refine_sr_noviewpred'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              #'train_occ',   # train the occs
                              #'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']

exps['train_0715_refine_sr_occ_view_fixview'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']

groups['load_train_0715_refine_sr_occ_view_fixview'] = [

                # 'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/train_0715_refine_sr_occ_view_fixview/\'}'
                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/train_0715_refine_success_rate_occ_view/\'}'
                ]



exps['test_train_0715_refine_sr_occ_view_fixview_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_1030_franka2',
                              'load_train_0715_refine_sr_occ_view_fixview',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '50k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              'test_mode',
                              'fatest_validate',
                              #'quicker_snap',
                              'lr12', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              #'validate',    # do the nearest neighbor vis
                              'fastest_logging']


exps['train_0715_refine_sr_occ_view_veggie'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_veggie',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'quick_snap',
                              #'load_train_viewpred_occ2',
                              '41k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'test_mode',
                              #'fatest_validate',
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']



exps['finetune_train_veggie_debug'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_veggie_debug',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'quick_snap',
                              #'load_train_viewpred_occ2',
                              '41k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'test_mode',
                              #'fatest_validate',
                              #'quicker_snap',
                              'lr4', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']


exps['finetune_train_0317_refine_sr_occ_view_fixview_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_0317_franka3_traineval',
                              'load_train_0715_refine_sr_occ_view_fixview',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'quick_snap',
                              #'load_train_viewpred_occ2',
                              '41k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'test_mode',
                              #'fatest_validate',
                              #'quicker_snap',
                              'lr4', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']


exps['finetune_train_0317_refine_nonfixcrop_sr_occ_view_fixview_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_0317_franka3_traineval',
                              'load_train_0715_refine_sr_occ_view_fixview',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'quick_snap',
                              #'load_train_viewpred_occ2',
                              '45k_iters', # num_iters
                              'is_refine_net',
                              #'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'test_mode',
                              #'fatest_validate',
                              #'quicker_snap',
                              'lr4', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']


exps['finetune_train_0317_refine2_sr_occ_view_fixview_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_0317_franka2_traineval',
                              'load_train_0715_refine_sr_occ_view_fixview',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'quick_snap',
                              #'load_train_viewpred_occ2',
                              '45k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'test_mode',
                              #'fatest_validate',
                              #'quicker_snap',
                              'lr4', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']

exps['finetune_train_0317_merge_refine_sr_occ_view_fixview_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_0317_franka_merge_traineval',
                              'load_train_0715_refine_sr_occ_view_fixview',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'quick_snap',
                              #'load_train_viewpred_occ2',
                              '45k_iters', # num_iters
                              'is_refine_net',
                              # 'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'test_mode',
                              #'fatest_validate',
                              #'quicker_snap',
                              'lr4', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']

exps['plate_0612'] = ['mujoco_offline_metric',  # mode
                              'plate_0612',
                              'cluster_26',
                              'quick_snap',
                              '45k_iters',   # num_iters
                              'is_refine_net',
                              "metric_learning_loss_type_sr",
                              'lr4',         # 1e-3
                              'B2',          # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']

groups['load_plate_0612_r2'] = ['loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/plate_0612_r2/\'}']

exps['plate_0612_load'] = ['mujoco_offline_metric',  # mode
                              'plate_0612',
                              'load_plate_0612_r2',
                              'cluster_26',
                              'quick_snap',
                              '45k_iters',   # num_iters
                              'is_refine_net',
                              "metric_learning_loss_type_sr",
                              'lr4',         # 1e-3
                              'B2',          # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']

exps['finetune_train_0715_refine_sr_occ_view_fixview_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_1030_franka3_traineval',
                              'load_train_0715_refine_sr_occ_view_fixview',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'quick_snap',
                              #'load_train_viewpred_occ2',
                              '41k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'test_mode',
                              #'fatest_validate',
                              #'quicker_snap',
                              'lr4', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'faster_logging']


groups['load_finetune_train_0715_refine_sr_occ_view_fixview_franka'] = [

                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/finetune_train_0715_refine_sr_occ_view_fixview_franka14_l4_novp_41k2/\'}'
                ]

exps['test_finetune_train_0715_refine_sr_occ_view_fixview_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_1030_franka3_traineval',
                              'load_finetune_train_0715_refine_sr_occ_view_fixview_franka',
                              #'load_train_0715_refine_sr_occ_view_fixview',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'quick_snap',
                              #'load_train_viewpred_occ2',
                              '41k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              'test_mode',
                              'fatest_validate',
                              #'quicker_snap',
                              'lr12', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              #'validate',    # do the nearest neighbor vis
                              'faster_logging']

groups['load_finetune_train_0715_refine_sr_occ_view_fixview_franka'] = [

                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/finetune_train_0715_refine_sr_occ_view_fixview_franka_cam14_l4_novp_41k2/\'}'
                ]


groups['load_train_0715_det_refine_sr_occ_view_fixview'] = [

                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/train_0715_refine_sr_occ_view_fixview/\'}'
                ]

exps['test_finetune_train_0715_refine_sr_occ_view_fixview_franka'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_1030_franka3_traineval',
                              'load_finetune_train_0715_refine_sr_occ_view_fixview_franka',
                              #'load_train_0715_refine_sr_occ_view_fixview',
                              #'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '50k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              'test_mode',
                              'fatest_validate',
                              #'quicker_snap',
                              'lr12', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              #'validate',    # do the nearest neighbor vis
                              'fastest_logging']



exps['train_0715_det_refine_sr_occ_view_fixview'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_det_merge',
                              'load_train_0715_det_refine_sr_occ_view_fixview',
                              'cluster_30',
                              #'load_train_viewpred_occ2',
                              '80k_iters', # num_iters
                              'is_refine_net',
                              'fix_crop',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'lr12', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']


groups['load_train_0715_refine_success_rate_occ_view'] = [

                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/train_0715_refine_success_rate_occ_view/\'}'
                ]

exps['train_0715_refine_success_rate_occ_view_conti'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              #'grasptop_all_0629_data',
                              'grasp_c6_r3_0715_merge',
                              'cluster_30',
                              'load_train_0715_refine_success_rate_occ_view',
                              #'load_train_viewpred_occ2',
                              '80k_iters', # num_iters
                              'is_refine_net',
                              "metric_learning_loss_type_sr",
                              #'quicker_snap',
                              'lr4', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']

exps['train_0629_refine_debug'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_0629_data',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              'quickest_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              #'train_occ',   # train the occs
                              #'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']
groups['load_train_0629_refine_debug'] = [
                'total_init = True',
                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE_METRIC/train_0629_refine_debug/\'}'
                ]

exps['train_0629_refine_debug_load'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_0629_data',
                              'load_train_0629_refine_debug',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              'quickest_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              #'train_occ',   # train the occs
                              #'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']


exps['train_0629_viewpred_occ'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_0629_data',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']

exps['train_0629_viewpred_occ_refine'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_0629_data',
                              #'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              'is_refine_net',
                              #"is_init_cluter_with_instance",
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']




groups['load_train_viewpred_occ2'] = [
                'total_init = True',
                'loadname = {\'model\':\'./checkpoints/MUJOCO_OFFLINE/train_viewpred_occ2/\'}'
                ]


exps['test_train_viewpred_occ2'] = ['mujoco_offline_metric',  # mode
                              #'mujoco_offline_data',  # input format
                              'grasptop_all_data',
                              'load_train_viewpred_occ2',
                              '40k_iters', # num_iters
                              "create_cluster_center",
                              #'quicker_snap',
                              'lr3', # 1e-3
                              'B2',  # batch_size
                              'train_feat',  # train the feats
                              'train_occ',   # train the occs
                              'train_view',  # train the view
                              'validate',    # do the nearest neighbor vis
                              'slow_logging']


exps['train_quant_vis_viewpred_occ_dt1542020'] = ['mujoco_offline_metric',  # mode
                                        'mujoco_offline_data_dt1542020',  # input format
                                        '40k_iters', # num_iters
                                        'lr3', # 1e-3
                                        'B2',  # batch_size
                                        'train_feat',  # train the feats
                                        'train_occ',   # train the occs
                                        'train_view',  # train the view
                                        'validate',    # do the nearest neighbor vis
                                        'faster_logging']

exps['quant_generate_data'] = ['mujoco_offline_metric',
                               'mujoco_offline_data_generate_data',
                               '100_iters',  # because there are only 10 files in the valset
                               'B1',   # just want to make life easier
                               'frozen_feat',  # freeze the featnet
                               'frozen_occ',   # freeze the occnet
                               'frozen_view',  # freeze the viewnet
                               'generate_data',  # do do_generate_data = True
                               'fastest_logging']

exps['train_metric_learner'] = ['mujoco_offline_metric',
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
groups["metric_learning_loss_type_sr"] = ['metric_learning_loss_type = "success_rate"']
groups["is_refine_net"] = ["is_refine_net = True"]
groups["is_init_cluter_with_instance"] = ["is_init_cluter_with_instance = True"]

groups["create_cluster_center"] = ["do_compute_cluster_center = True"]
groups["top_grasp_only"] = ["top_grasp_only = True"]

groups['cluster_30'] = ['max_clusters = 30']
groups['cluster_26'] = ['max_clusters = 26']

# featnet
groups['train_feat'] = ['do_feat = True',
                        'feat_dim = 64',
]

groups['train_feat64'] = ['do_feat = True',
                        'feat_dim = 64',
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
groups["test_mode"] = ["backprop_on_train = False"]
# moco-net
groups['train_moc'] = ['do_moc = True',
                       'dict_len = 100000',
                       'num_neg_samples = 2048',
                       'do_bn = False',
                       'num_pos_samples = 1024',
                       'emb_3D_mindist = 8.0']

groups['validate'] = ['do_validation = True',
                      'validate_after = 50']

groups['fatest_validate'] = ['do_validation = True',
                      'validate_after = 1']

#groups['validate_dt1542020.txt'] = ['do_validation = True',
#                      'validation_path = "backend/train_files/train_val_file_dt1542020.txt"',
#                      'validate_after = 100']

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
                                 'valset = "parsed_task_grasptop_all_val"',
                                 'testset = "parsed_task_grasptop_all_val"',
                                 'dataset_list_dir = "/projects/katefgroup/quantized_policies/data/grasptop"',
                                 'dataset_format = "txt"'
]

groups['grasptop_all_0629_data'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "parsed_task_grasptop_all_0629_train"',
                                 'valset = "parsed_task_grasptop_all_0629_val"',
                                 'testset = "parsed_task_grasptop_all_0629_val"',
                                 'dataset_list_dir = "/projects/katefgroup/quantized_policies/data/grasptop"',
                                 'dataset_format = "txt"'
]

groups['grasp_c6_r3_0715_merge'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_c6_r3_0715_merge_train"',
                                 'valset = "grasp_c6_r3_0715_merge_val"',
                                 'testset = "grasp_c6_r3_0715_merge_val"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]

groups['grasp_c6_r3_0715_st_merge'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_c6_r3_0715_train_st_merge_train"',
                                 'valset = "grasp_c6_r3_0715_merge_val"',
                                 'testset = "grasp_c6_r3_0715_merge_val"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]


groups['grasp_1030_franka'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_franka_val"',
                                 'valset = "grasp_franka_val"',
                                 'testset = "grasp_franka_val"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]

groups['grasp_1030_franka2'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_franka2_val"',
                                 'valset = "grasp_franka2_val"',
                                 'testset = "grasp_franka2_val"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]

groups['grasp_1030_franka3'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_franka3_val"',
                                 'valset = "grasp_franka3_val"',
                                 'testset = "grasp_franka3_val"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]


groups['grasp_veggie_debug'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "pnp_1212_debug"',
                                 'valset = "pnp_1212_debug"',
                                 'testset = "pnp_1212_debug"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]


groups['grasp_veggie'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "pnp_1212_train"',
                                 'valset = "pnp_1212_val"',
                                 'testset = "pnp_1212_val"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]

groups['grasp_1030_franka3_traineval'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_franka3_split_train"',
                                 'valset = "grasp_franka3_split_test"',
                                 'testset = "grasp_franka3_split_test"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]


groups['grasp_0317_franka3_traineval'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_franka_iccv_train"',
                                 'valset = "grasp_franka_iccv_test"',
                                 'testset = "grasp_franka_iccv_test"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]


groups['grasp_0317_franka2_traineval'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_franka_iccv_train2"',
                                 'valset = "grasp_franka_iccv_test2"',
                                 'testset = "grasp_franka_iccv_test2"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]


groups['grasp_0317_franka_merge_traineval'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_franka_merge_train"',
                                 'valset = "grasp_franka_merge_test"',
                                 'testset = "grasp_franka_merge_test"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]

groups['plate_0612'] = ['dataset_name = "mujoco_offline"',
                        'H = %d' % H,
                        'W = %d' % W,
                        'trainset = "train_plate"',
                        'valset = "test_plate"',
                        'testset = "test_plate"',
                        'dataset_list_dir = "plate"',
                        'dataset_format = "txt"'
]


groups['grasp_c6_r3_0715_st1_merge'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_c6_r3_0715_train_st1_merge_train"',
                                 'valset = "grasp_c6_r3_0715_merge_val"',
                                 'testset = "grasp_c6_r3_0715_merge_val"',
                                 'dataset_list_dir = "grasp"',
                                 'dataset_format = "txt"'
]

groups['grasp_c6_r3_0715_det_merge'] = ['dataset_name = "mujoco_offline"',
                                 'H = %d' % H,
                                 'W = %d' % W,
                                 'trainset = "grasp_c6_r3_0715_merge_train"',
                                 'valset = "grasp_c6_r3_val_0715_det_merge_train"',
                                 'testset = "grasp_c6_r3_val_0715_det_merge_train"',
                                 'dataset_list_dir = "grasp"',
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
