#python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_viewpred_occ --run_name=1
#python -W ignore main.py MUJOCO_OFFLINE --exp_name=test_train_viewpred_occ2 --run_name=test_train_viewpred_occ2



#train_viewpred_occ
#CUDA_VISIBLE_DEVICES=0 python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_viewpred_occ_0629 --run_name=train_viewpred_occ_0629_nn_test

#python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_viewpred_occ_0629 --run_name=train_viewpred_occ_0629_nn_nonorm
#python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_viewpred_occ_0629 --run_name=train_viewpred_occ_0629_nn_nonorm_test


#*** new dataset
#CUDA_VISIBLE_DEVICES=0 python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_viewpred_occ_0715_c6_r3 --run_name=train_viewpred_occ_0715_c6_r3

#python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_viewpred_occ_push_0723 --run_name=train_viewpred_occ_push_0723


#train_metric leanring
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0629 --run_name=train_0629_init_cluster_center
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0629_refine --run_name=train_0629_refine3

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0629_refine_debug_load --run_name=train_0629_refine_debug_load

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0629_viewpred_occ_refine --run_name=train_0629_viewpred_occ_refine_no_init
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0629_viewpred_occ_refine --run_name=train_0629_viewpred_occ_refine_no_init2

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0629_viewpred_occ --run_name=train_0629_viewpred_occ


## 2d baseline
#python -W ignore main.py MUJOCO_OFFLINE_METRIC_2D --exp_name=train_2d_0715 --run_name=train_2d_0715_2
#python -W ignore main.py MUJOCO_OFFLINE_METRIC_2D --exp_name=train_2d_0715_sr --run_name=train_2d_0715_sr_2

#*** new dataset
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_refine --run_name=train_0715_refine

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_refine_success_rate --run_name=train_0715_refine_success_rate4

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_refine_success_rate_occ_view --run_name=train_0715_refine_success_rate_occ_view


#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_refine_success_rate_occ_view_conti --run_name=train_0715_refine_success_rate_occ_view_conti
 
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_refine_sr_occ_view_fixview  --run_name=0316_train_0715_refine_sr_occ_view_fixview # fix crop


#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_refine_sr_noviewpred --run_name=0316_train_0715_refine_sr_noviewpred_2 # fix crop

# evaluate on models with detector
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_det_refine_sr_occ_view_fixview  --run_name=train_0715_det_refine_sr_occ_view_fixview # fix crop


# train with one-trial approximate on success rate
#CUDA_VISIBLE_DEVICES=3 python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_st_refine_sr_occ_view_fixview  --run_name=train_0715_st_refine_sr_occ_view_fixview # fix crop

#CUDA_VISIBLE_DEVICES=3 python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_st1_refine_sr_occ_view_fixview  --run_name=train_0715_st1_refine_sr_occ_view_fixview_0316 # fix crop



#CUDA_VISIBLE_DEVICES=3 python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_refine_sr_occv_fixcrop_top  --run_name=train_0715_refine_sr_occv_fixcrop_top



#CUDA_VISIBLE_DEVICES=3 python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_rgb_occ_det_as2 --run_name=train_rgb_occ_det_as2

# test on franka data

#CUDA_VISIBLE_DEVICES=3 python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=test_train_0715_refine_sr_occ_view_fixview_franka --run_name=test_train_0715_refine_sr_occ_view_fixview_franka # fix crop


#CUDA_VISIBLE_DEVICES=3 python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_0715_refine_sr_occ_view_fixview_franka --run_name=finetune_train_0715_refine_sr_occ_view_fixview_franka_cam14_l4_novp_41k2
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=test_finetune_train_0715_refine_sr_occ_view_fixview_franka --run_name=test_finetune_train_0715_refine_sr_occ_view_fixview_franka_cam14_l4_novp_41k2
#_novp # fix crop


#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_0317_refine_sr_occ_view_fixview_franka --run_name=finetune_train_0317_refine_sr_occ_view_fixview_franka

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_0317_refine_nonfixcrop_sr_occ_view_fixview_franka --run_name=finetune_train_0317_refine_nonfixcrop_sr_occ_view_fixview_franka

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_0317_refine2_sr_occ_view_fixview_franka --run_name=finetune_train_0317_refine2_sr_occ_view_fixview_franka

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_0317_merge_refine_sr_occ_view_fixview_franka --run_name=finetune_train_0317_merge_refine2_sr_occ_view_fixview_franka

# python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_0317_merge_refine_sr_occ_view_fixview_franka --run_name=finetune_train_0317_merge_refine2_nonfixcrop_sr_occ_view_fixview_franka_45k

# python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_0317_refine_nonfixcrop_sr_occ_view_fixview_franka --run_name=finetune_train_0317_refine_nonfixcrop_sr_occ_view_fixview_franka


python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=plate_0612_load --run_name=plate_0612_r2_cluttered
#python -W ignore main.py MUJOCO_OFFLINE_METRIC_2D --exp_name=plate_0612_load --run_name=train_2d_0827

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=test_finetune_train_0715_refine_sr_occ_view_fixview_franka --run_name=test_finetune_train_0715_refine_sr_occ_view_fixview_franka_cam14_l4_novp_41k2

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=test_train_0715_refine_sr_occv_fixcrop_top_franka3  --run_name=test_train_0715_refine_sr_occv_fixcrop_top_franka3 # fix crop
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_0715_refine_sr_occv_fixcrop_top_franka3  --run_name=finetune_train_0715_refine_sr_occv_fixcrop_top_franka3 # fix crop

#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=train_0715_refine_sr_occ_view_veggie --run_name=train_0715_refine_sr_occ_view_veggie
#python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=finetune_train_veggie_debug --run_name=finetune_train_veggie_debug
