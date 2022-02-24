# with trained policies as init
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top.yaml --task_config_file tasks/grasptop/task_grasptop_debug.yaml  --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_debug.yaml

# with controller as init
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_controller.yaml --task_config_file tasks/grasptop/task_grasptop_rescale_small.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_rescale_small.yaml

# all objects
## generate data for top grasp
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_controller.yaml --task_config_file tasks/grasptop/task_grasptop_all_train.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_train3.yaml
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_controller.yaml --task_config_file tasks/grasptop/task_grasptop_all_test.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_test3.yaml --run_name=test_parsing





# controller that considers 6d pos
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasptop/task_grasptop_all_train.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_train3.yaml
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasptop/task_grasptop_all_train_orn.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_train3.yaml


#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasptop/car.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_car.yaml --run_name car

# assignment
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasptop/task_grasptop_all_train.yaml --output_file tasks/parsed_output/grasp/tmp.yaml --run_name train_split1


#python compression/policy_compression_video.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge_video.yaml --output_file tasks/parsed_output/grasp/tmp.yaml --run_name train_split1



#python compression/policy_compression_with_init_policies_3dtensor.py compression/configs/grasp_to_posorn_with_selector.yaml --task_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge_video.yaml --output_file tasks/parsed_output/grasp/tmp.yaml --run_name train_split1


#compression/configs/grasp_from_top_controller_with_selector.yaml 


python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split1_conti.yaml  --run_name train_split1
# done starting 0
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split1.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split1_conti.yaml  --run_name train_split1
#ok c python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split1_50.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split1_50_cont1.yaml  --run_name train_split1_50
#ok python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split1_100.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split1_100.yaml  --run_name train_split1_100
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split1_150.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split1_150_cont2.yaml  --run_name train_split1_150


# starting 200
# v python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split2.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split2.yaml  --run_name train_split2
#$ c ~290
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split2_50.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split2_50_conti2.yaml  --run_name train_split2_50
# cv python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split2_100.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split2_100_cont1.yaml  --run_name train_split2_100
# v python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split2_150.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split2_150.yaml  --run_name train_split2_150


# starting 400
#$ cc
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split3.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split3_conti2.yaml  --run_name train_split3

#~~~~484-500
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split3_50.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split3_50_cont1.yaml  --run_name train_split3_50


#v python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split3_100.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split3_100.yaml  --run_name train_split3_100
##41 python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_train_0715_split3_150.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_split3_150.yaml  --run_name train_split3_150


##73 python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val.yaml  --run_name val
##35 python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715_split50.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val_50.yaml  --run_name val_50


#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn_det.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val_det.yaml  --run_name val

#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn_det.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val_det_252.yaml  --run_name val252

#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn_det.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715_split100.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val_det_split100_4.yaml  --run_name val100
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn_det.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715_split150.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val_det_split150_4.yaml  --run_name val2
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn_det.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715_split50.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val_det_split50.yaml  --run_name val


# 125-150
# r 
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715_split100.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val_100_conti.yaml  --run_name val_100
# 195-161 
#$ c 
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_val_0715_split150.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_val_150_conti.yaml  --run_name val_150



##56 python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_test_0715.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_test.yaml  --run_name test


# todo
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_test_0715_split100.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_test_100.yaml  --run_name test_100
#python compression/policy_compression_assignment.py compression/configs/grasp_to_posorn.yaml --task_config_file tasks/grasp/grasp_c6_r3_test_0715_split150.yaml --output_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_test_150.yaml  --run_name test_150




#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_controller.yaml --task_config_file tasks/grasptop/task_grasptop_center_working.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_tmp.yaml
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_controller.yaml --task_config_file tasks/grasptop/task_grasptop_rim_working.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_tmp.yaml


#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_controller.yaml --task_config_file tasks/grasptop/task_grasptop_tmp.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_tmp.yaml
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_controller.yaml --task_config_file tasks/grasptop/task_grasptop_all_train.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_train.yaml
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_controller.yaml --task_config_file tasks/grasptop/task_grasptop_all_test.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_test.yaml


# with controller as init and with selector
#python compression/policy_compression_with_init_policies_3dtensor.py compression/configs/grasp_from_top_controller_with_selector.yaml --task_config_file tasks/grasptop/task_grasptop_rescale_small.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_rescale_small.yaml


# for testing center controller from top
#python compression/policy_compression_with_init_policies.py compression/configs/grasp_from_top_center_controller.yaml --task_config_file tasks/grasptop/task_grasptop_center_small.yaml --output_file tasks/parsed_output/grasptop/parsed_task_grasptop_center_small.yaml
