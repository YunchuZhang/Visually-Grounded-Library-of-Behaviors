# render yaml files given a folder
import os
import yaml
import numpy as np
import random

import pcp_utils


def rand_sample(range_x):
    if "#" in range_x:
        x_min, x_max = [float(x) for x in range_x.split("#")]
        return x_min + np.random.rand(1)[0] * (x_max - x_min)
    if "," in range_x:
        possible_numbers = [float(x) for x in range_x.split(",")]
        return random.choice(possible_numbers)


    return float(range_x)




def euler_rand_sample(euler):
    x, y, z = euler.split(" ")
    sample_x = rand_sample(x)
    sample_y = rand_sample(y)
    sample_z = rand_sample(z)

    if isinstance(sample_x, np.float64):
        sample_x = sample_x.item()

    if isinstance(sample_y, np.float64):
        sample_y = sample_y.item()

    if isinstance(sample_z, np.float64):
        sample_z = sample_z.item()

    return [sample_x, sample_y, sample_z]


"""
objects not possible to grasp:
  # ** cup cake
  bowls_64d7f5eb886cfa48ce6101c7990e61d
dog plate
#  bowls_3f6a6718d729b77bed2eab6efdeec5f
# bowl with flat top
  bowls_6501113f772fc798db649551c356c6e


"""


# getting train/test split

mesh_xml_dir = pcp_utils.utils.get_object_xml_dir()

categories_folder = ["bus", "can", "cups", "bowls", "car", "bottles"]
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

scales = ["0.7#1.2"]
euler_xyz = [ "0 0 0#3.14"]
#categories_folder = ["bus", "can", "cups", "bowls"]
#scales = ["1#1.5" "0.5#1", "0.8,1.2,1.5", "0.7,1.2,1.5"]
nrun_each_object = 2
# run
#euler_xyz = [ "0 0 0#3.14", "0,1.57 0,1.57 0#3.14", "0 0 0#3.14",  "0 0 0#3.14"]
output_folder = "tasks/grasp"
output_file_prefix = "grasp_c6_tmp"


exclude_files = {"cups": ["64d7f5eb886cfa48ce6101c7990e61d.xml", "3f6a6718d729b77bed2eab6efdeec5f.xml"], "car": ["365af82203c96fbca92a4cdad802b4", "37a7f79b9eeb5096b11647ffa430660"]}
train_f = open(os.path.join(output_folder, output_file_prefix  + "_train.txt"), "w")
val_f = open(os.path.join(output_folder, output_file_prefix  + "_val.txt"), "w")
test_f = open(os.path.join(output_folder, output_file_prefix  + "_test.txt"), "w")

for category_id, category_folder in enumerate(categories_folder):
    if category_folder in exclude_files:
        files_to_exclude = exclude_files[category_folder]
    else:
        files_to_exclude = []
    all_files = os.listdir(os.path.join(mesh_xml_dir, category_folder))

    nfiles = len(all_files)
    n_train_files = int(nfiles * train_ratio)
    n_val_files = int(nfiles * val_ratio)
    n_test_files = nfiles - n_train_files - n_val_files

    np.random.shuffle(all_files)

    train_files = all_files[:n_train_files]
    val_files = all_files[n_train_files:n_train_files + n_val_files]
    test_files = all_files[n_train_files + n_val_files:]


    for train_file in train_files:
        if train_file not in files_to_exclude:
            train_f.writelines(category_folder + "/" + train_file + '\n')


    for val_file in val_files:
        if val_file not in files_to_exclude:
            val_f.writelines(category_folder + "/" + val_file + '\n')


    for test_file in test_files:
        if test_file not in files_to_exclude:
            test_f.writelines(category_folder + "/" + test_file + '\n')

train_f.close()
val_f.close()
test_f.close()

