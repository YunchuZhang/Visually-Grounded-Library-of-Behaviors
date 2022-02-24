#python compression/grasp_using_6d_graspnet.py compression/configs/grasp_6d_graspnet.yaml --task_config_file tasks/grasptop/task_grasptop_rescale_small.yaml

# run eval using trained 6dpos grasp
#python compression/policy_compression_assignment.py compression/configs/grasp_6dgrasp.yaml --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge.yaml --output_file tasks/parsed_output/grasp/6dgrasp_out.yaml  --run_name 6dof_grasp


# collect data to retrain 6dpos grasp
#python tests/collect_grasping_data.py compression/configs/grasp_6dgrasp_collect.yaml --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_merge.yaml --output_file tasks/parsed_output/tmp


#tasks/parsed_output/grasp/6dgrasp_val_0715.yaml --run_name 6dof_grasp

#tasks/grasp/grasp_c6_r3_test_0715.yaml 

# visualization
#python tests/plot_grasps.py --file_path vae_generated_grasps/grasps_899af991203577f019790c8746d79a6f.npy
#python tests/plot_grasps.py --file_path vae_generated_grasps/grasps_159e56c18906830278d8f8c02c47cde0_18.npy

#python tests/plot_grasps.py --file_path vae_generated_grasps/grasps_fn_159e56c18906830278d8f8c02c47cde0_15.npy
#




# run dexnet grasping
#python compression/policy_compression_assignment.py compression/configs/grasp_dexnet_collect.yaml --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge_video.yaml --output_file tasks/parsed_output/tmp

# test eval (comment out save_video in the class)
#python compression/policy_compression_assignment.py compression/configs/grasp_dexnet_collect.yaml --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge.yaml --output_file tasks/parsed_output/tmp2

# test eval for grasp6d
#python compression/policy_compression_assignment.py compression/configs/grasp_6dgrasp.yaml  --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge_video.yaml --output_file tasks/parsed_output/tmp
#python compression/policy_compression_assignment.py compression/configs/grasp_6dgrasp.yaml  --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge.yaml --output_file tasks/parsed_output/tmp

# collecting training samples for dexnet
#python compression/policy_compression_assignment.py compression/configs/grasp_dexnet_collect_data.yaml --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_merge.yaml --output_file tasks/parsed_output/tmp2


# collecting training samples for graspnet
#python compression/policy_compression_assignment.py compression/configs/grasp_6dgrasp_collect_data.yaml  --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_merge.yaml --output_file tasks/parsed_output/tmp --run_name=collect20_gan1

# collect object-specific data
#python compression/policy_compression_assignment.py compression/configs/grasp_6dgrasp_collect_data.yaml  --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_mug.yaml --output_file tasks/parsed_output/tmp --run_name=collect20_mug
#python compression/policy_compression_assignment.py compression/configs/grasp_6dgrasp_collect_data_fv.yaml  --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_mug.yaml --output_file tasks/parsed_output/tmp --run_name=collect20_mug2


python compression/policy_compression_assignment.py compression/configs/grasp_6dgrasp_collect_data2.yaml  --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge.yaml --output_file tasks/parsed_output/tmp --run_name=vae1
#python compression/policy_compression_assignment.py compression/configs/grasp_6dgrasp_collect_data2.yaml  --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_mug.yaml --output_file tasks/parsed_output/tmp --run_name=vae1
