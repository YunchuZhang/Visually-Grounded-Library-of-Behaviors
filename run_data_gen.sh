# python data_gen_collect.py env_xmls [--object_config_file obj_config_file ==> this should include class name]

#python data_gen/collect.py  data_gen/configs/data_gen_env.yaml  --object_config_file data_gen/configs/data_gen_objs.yaml

## for debugging
#python data_gen/collect.py  data_gen/configs/data_gen_env.yaml  --object_config_file tasks/parsed_output/grasptop/parsed_one_obj.yaml --data_name grasptop/parsed_one_obj


## for all the objects
#python data_gen/collect2.py  data_gen/configs/data_gen_env_new.yaml  --object_config_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_train.yaml --data_name grasptop/parsed_task_grasptop_all_0630
#python data_gen/collect2.py  data_gen/configs/data_gen_env_new_test.yaml  --object_config_file tasks/parsed_output/grasptop/parsed_task_grasptop_all_test.yaml --data_name grasptop/parsed_task_grasptop_all_0630 --run_name=test_gen



# testing new orn. init function, and scale geom objects
#python data_gen/collect2.py data_gen/configs/data_gen_env_new.yaml --object_config_file tasks/grasptop/task_grasptop_all_train_orn.yaml --data_name tmp

#python data_gen/collect2.py data_gen/configs/data_gen_env_new.yaml --object_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_merge.yaml --data_name tmp


#python data_gen/collect2.py data_gen/configs/data_gen_posorn.yaml --object_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_merge.yaml --data_name grasp/grasp_c6_r3_train_0715_merge

#python data_gen/collect2.py data_gen/configs/data_gen_posorn.yaml --object_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_merge.yaml --data_name grasp/grasp_c6_r3_val_0715_merge



#python data_gen/collect2.py data_gen/configs/data_gen_posorn_det.yaml --object_config_file tasks/parsed_output/grasp/grasp_c6_r3_val_0715_val_det_merge.yaml --data_name grasp/grasp_c6_r3_val_0715_det_merge


# training object uses only one trial as successful rate
python data_gen/collect2.py data_gen/configs/data_gen_env_st.yaml --object_config_file tasks/parsed_output/grasp/grasp_c6_r3_train_0715_merge.yaml --data_name grasp/grasp_c6_r3_0715_train_st_merge --run_name gen_large









#python data_gen/collect.py  data_gen/configs/data_gen_env.yaml  --object_config_file tasks/parsed_output/grasptop/parsed_task_grasptop_all.yaml --data_name grasptop/parsed_task_grasptop_all2