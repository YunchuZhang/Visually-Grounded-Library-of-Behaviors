# render yaml files given a folder
import os
import yaml
import numpy as np
import random

import pcp_utils

random.seed(0)



def rand_sample(range_x):
    if "#" in range_x:
        x_min, x_max = [float(x) for x in range_x.split("#")]
        return x_min + np.random.rand(1)[0] * (x_max - x_min)
    if "," in range_x:
        possible_numbers = [float(x) for x in range_x.split(",")]
        return np.array(random.choice(possible_numbers), np.float32)

    return np.array(range_x, np.float32)




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


def main1():

    mesh_xml_dir = pcp_utils.utils.get_object_xml_dir()
    
    categories_folder = ["car"]
    scales = ["0.7#1.2"]
    euler_xyz = [ "0 0 0#3.14"]
    #categories_folder = ["bus", "can", "cups", "bowls"]
    #scales = ["1#1.5" "0.5#1", "0.8,1.2,1.5", "0.7,1.2,1.5"]
    nrun_each_object = 2
    # run
    #euler_xyz = [ "0 0 0#3.14", "0,1.57 0,1.57 0#3.14", "0 0 0#3.14",  "0 0 0#3.14"]
    output_folder = "tasks/grasptop"
    
    
    
    for category_id, category_folder in enumerate(categories_folder):
    
        scale_range = scales[category_id]#[float(x) for x in scales[category_id].split("#")]
        yaml_dict = dict()
        yaml_dict['objs'] = dict()
    
        for obj_xml in os.listdir(os.path.join(mesh_xml_dir, category_folder)):
    
            for run_id in range(nrun_each_object):
               obj_name = category_folder + "_" + obj_xml[:-5] + "_run" + str(run_id)
               yaml_dict['objs'][obj_name] = dict()
               yaml_dict['objs'][obj_name]["obj_xml_file"] = os.path.join(category_folder, obj_xml)
               yaml_dict['objs'][obj_name]["scale"] = rand_sample(scale_range).item() #(scale_range[0] + np.random.rand(1)[0] * (scale_range[1] - scale_range[0])).item()
               yaml_dict['objs'][obj_name]['mass'] = 1.2
               yaml_dict['objs'][obj_name]['euler_xyz'] = " ".join([str(x) for x in euler_rand_sample(euler_xyz[category_id])])
    
               print("hello")
    
    
    
        with open(os.path.join(output_folder, category_folder + ".yaml"), 'w') as file:
            yaml.dump(yaml_dict, file)
    
     	#yaml_dict['objs'][obj_xml] 




def main2():

    # read from the train test split files
    
    object_list_file = "tasks/grasp/grasp_c6_test.txt"
    output_filename= "tasks/grasp/grasp_c6_r3_test_0715.yaml"


    mesh_xml_dir = pcp_utils.utils.get_object_xml_dir()
    
    categories_folder = ["car", "bus", "can", "cups", "bowls", "bottles"]
    category_to_id = dict()

    for category_id, category in enumerate(categories_folder):
    	category_to_id[category] = category_id


    #scales = ["0.7#1.2"]
    #euler_xyz = [ "0 0 0#3.14"]
    #categories_folder = ["bus", "can", "cups", "bowls"]
    scales = ["1#1.5", "0.5#1", "0.8,1.2,1.5", "0.7,1.2,1.5", "0.7#1.2", "1.0"]
    nrun_each_object = 3
    # run
    euler_xyz = [ "0 0 0#3.14", "0,1.57 0,1.57 0#3.14", "0 0 0#3.14",  "0 0 0#3.14", "0 0 0#3.14", "0,1.57 0,1.57 0#3.14"]
    
    assert(len(scales) == len(categories_folder))
    assert(len(euler_xyz) == len(categories_folder))
    yaml_dict = dict()
    yaml_dict['objs'] = dict()
    
    obj_id = 0

    with open(object_list_file, 'r') as f:
        for filename in f:
            category_name, obj_xml =  filename.split("/")
            obj_xml = obj_xml.strip()

            category_id = category_to_id[category_name]
            scale_range = scales[category_id]#[float(x) for x in scales[category_id].split("#")]

            for run_id in range(nrun_each_object):
               obj_name =  "id{:04d}".format(obj_id) + f"_{category_name}_{obj_xml[:-5]}_run{run_id}"
               obj_id += 1
               print(obj_name)
               yaml_dict['objs'][obj_name] = dict()
               yaml_dict['objs'][obj_name]["obj_xml_file"] = os.path.join(category_name, obj_xml)
               yaml_dict['objs'][obj_name]["scale"] = rand_sample(scale_range).item() #(scale_range[0] + np.random.rand(1)[0] * (scale_range[1] - scale_range[0])).item()
               yaml_dict['objs'][obj_name]['mass'] = 1.2
               yaml_dict['objs'][obj_name]['euler_xyz'] = " ".join([str(x) for x in euler_rand_sample(euler_xyz[category_id])])
    print(f"generate a total of {obj_id} number of objects")
    
    with open(output_filename, 'w') as file:
        yaml.dump(yaml_dict, file)

if __name__=="__main__":
    main2()
