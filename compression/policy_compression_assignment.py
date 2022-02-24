
import pcp_utils
import sys
import os
import click
import yaml
import random
RANDOM_SEED = 0


# add gym and baseline to the dir
gym_path = pcp_utils.utils.get_gym_dir()
baseline_path = pcp_utils.utils.get_baseline_dir()
sys.path.append(gym_path)
sys.path.append(baseline_path)


# make symbolic link of the mesh under quantize-gym/gym/envs/robotics/assets/stls
source_mesh_dir = pcp_utils.utils.get_mesh_dir()
gym_mesh_dir = os.path.join(pcp_utils.utils.get_gym_dir(), 'gym/envs/robotics/assets/stls/meshes')


if not os.path.exists(gym_mesh_dir):
    os.symlink(source_mesh_dir, gym_mesh_dir,  target_is_directory=True)



import argparse
from pcp_utils import utils
from pcp_utils.mesh_object import MeshObject
from pcp_utils.utils import Config
from pcp_utils.parse_task_files import generate_integrated_xml

import tqdm
import numpy as np


##### Imports related to environment #############
import gym


class PolicyCompressor:
    class Config(Config):
        num_rollouts = 100
        max_path_length = 50
        accept_threshold = 0.9
        num_threads = 5
        bbox_indicator = False


        # params for the environment
        env_name = None
        env_base_xml_file = "" #table, the kind of robot, object placeholder
        n_robot_dof = 8 # 4 means xyz, 6 means having rotations
        place = False
        render = True
        randomize_color = False
        init_info = None


    def __init__(self,  config:Config, initial_policies=[], output_file="", run_name=""):

        self.config = config

        self.num_rollouts = config.num_rollouts
        self.max_path_length = config.max_path_length
        self.accept_threshold = config.accept_threshold
        self.num_threads = config.num_threads
        self.env_name = config.env_name
        self.env_base_xml_file = config.env_base_xml_file
        self.bbox_indicator = config.bbox_indicator
        self.n_robot_dof = config.n_robot_dof
        self.randomize_color = config.randomize_color
        self.init_info = config.init_info

        self.run_name = run_name

        self.policy_bank = initial_policies

        self.output_file = output_file
        output_folder = "/".join(output_file.split("/")[:-1])
        utils.makedir(output_folder)
        self.objects_output_xml = dict()
        self.object_output_xml_id = 0

        self.clusters = {}
        # key: name of the cluster(c1)
        # value:
        #        objects: objects inside this clusters  (objects, success rate)
        #        expert_id: expert associate to it
        #        expert_name:
        #        3d tensor model:

        self.object_to_cluster = [] #just a lookup
        self.num_clusters = 0
        self.object_not_clustered = []
        self.success_rates_over_class = dict()
        self.failed_object = []
        self.success_rates = []

        self.init_clusters()

    def init_clusters(self):
        for policy in self.policy_bank:
            cluster_id = self.num_clusters
            self.clusters[f'c{cluster_id}'] =dict()
            self.clusters[f'c{cluster_id}']['objects'] = []
            self.clusters[f'c{cluster_id}']['expert_name'] = policy.policy_name
            self.clusters[f'c{cluster_id}']['expert'] = policy #.policy_model
            self.num_clusters += 1


    # Access to a minimal policy bank, also has information about which meshes to run on which policy
    # Takes in a new object and determines if it can be merged with an existing policy or can spawn a new policy
    # input: new object that needs to be classified
    # mesh should be mesh_id like 159e56c18906830278d8f8c02c47cde0, or b9004dcda66abf95b99d2a3bbaea842a which are ShapeNet ids
    def add_object(self, obj):
        #share the env

        # make xml for the object:
        integrated_xml = generate_integrated_xml(self.env_base_xml_file, obj.obj_xml_file, scale=obj.scale, mass=obj.mass, euler=obj.euler,
                add_bbox_indicator=self.bbox_indicator, randomize_color=self.randomize_color, prefix=self.run_name)
        #obj.xml_file)# xml_path="fetch/pick_and_place_kp30000_debug.xml") #xml_path=obj.xml_file)
        env = gym.make(self.env_name, xml_path=integrated_xml, use_bbox_indicator=self.bbox_indicator,
            n_actions=self.n_robot_dof, init_info=self.init_info)

        env.seed(RANDOM_SEED)
        env.action_space.seed(RANDOM_SEED)
        print(f'max env steps are: {env._max_episode_steps}')

        # env.render()
        # this ordering should be something learnable
        is_clustered = False
        success_rates = np.zeros(len(self.clusters.items()))

        for cid, cluster in self.clusters.items():
            env.seed(RANDOM_SEED)
            # load policy of the first mesh (parent mesh) in an existing cluster
            print("Checking performance of {} on policy for {}: {}".format(obj.name, cid, cluster['expert'].policy_name))
            stats = cluster['expert'].run_forwards(env, obj=obj, num_rollouts=self.num_rollouts, path_length=self.max_path_length, render=self.config.render, cluster_name=cluster['expert'].policy_name, place=self.config.place)#, accept_threshold=self.accept_threshold)

            success_rate = stats['success_rate']
            print("Success Rate ", success_rate)

            success_rates[int(cid[1:])] = success_rate

            if success_rate >= self.accept_threshold: #* base_success_rate:

                self.object_to_cluster.append((obj, cid))

                cluster['objects'].append((obj, success_rate))
                is_clustered = True
            # if fail: randomly select from on of the top

        self.success_rates_over_class[obj.name] = " ".join([str(success_rates[x]) for x in range(self.num_clusters)])

        self.success_rates.append(success_rates)

        if not is_clustered:
            self.object_not_clustered.append(obj)
            self.object_to_cluster.append((obj, "not_assigned"))

        gym_xml_path = os.path.join(pcp_utils.utils.get_gym_dir(), 'gym/envs/robotics/assets')
        integrated_xml_full_path = os.path.join(gym_xml_path,  integrated_xml)
        os.remove(integrated_xml_full_path)

        print("failed objects")
        print(self.failed_object)


        env.close()


    def output_xml(self):
        # output cluster name

        clusters_output = dict()
        for cluster_name in self.clusters:
            clusters_output[cluster_name] = dict()
            clusters_output[cluster_name]['expert_name'] = self.clusters[cluster_name]['expert_name']
            clusters_output[cluster_name]['fn'] = self.clusters[cluster_name]['expert'].__class__.__module__ + ":" + \
                                                         self.clusters[cluster_name]['expert'].__class__.__name__
            clusters_output[cluster_name]['params'] = self.clusters[cluster_name]['expert'].config.__dict__


        output_clusters = dict()
        output_clusters['clusters'] = clusters_output

        output_objs = dict()
        # output objects
        for obj, cluster_id in self.object_to_cluster[self.object_output_xml_id:]:
            self.objects_output_xml[obj.name] = dict()
            for key_id, item in obj.config.__dict__.items():

                if isinstance(item, list):
                    item = " ".join([str(x) for x in item])
                #elif isinstance(item, )

                self.objects_output_xml[obj.name][key_id] = item
            self.objects_output_xml[obj.name]['cluster_id'] = cluster_id
            self.objects_output_xml[obj.name]['success_rates_over_class'] = self.success_rates_over_class[obj.name] 

        self.object_output_xml_id = len(self.objects_output_xml)
        output_objs["objs"] = self.objects_output_xml

        with open(self.output_file, 'w') as file:
            yaml.dump(self.config.__dict__, file, default_flow_style=False)
            yaml.dump(output_clusters, file, default_flow_style=False)
            yaml.dump(output_objs, file, default_flow_style=False)

        print("avg success rate:", np.mean(np.stack(self.success_rates, axis=0),axis=0))

    def print_policy_summary(self):
        # want to write out something with the policy and objects, and we can load from it.

        print("Compressed Meshes")
        for cid, cluster in self.clusters.items():
            print(f'cluster {cid}')
            expert_name = cluster['expert_name']
            print(f'expert used in the cluster: {expert_name}')
            for obj, success_rate in cluster['objects']:
                print(f'    {obj.name} ({success_rate})')
        print("==========================")
        for obj in self.object_not_clustered:
            print(f'    {obj.name} ')

        #print("Percentage compression for threshold {} : {} ".format(self.accept_threshold, self.num_clusters/len(MESHES)))
        #print("Cluster Accuracies")
        #print(self.cluster_acc)

@click.command()
@click.argument("config_file")#config
@click.option("--task_config_file") #for the objects
@click.option("--output_file") #for the objects
@click.option("--run_name") #define run name to avoid generate_xml to overwrite
def main(config_file, task_config_file, output_file, run_name):


    config = utils.config_from_yaml_file(config_file)
    # build detector
    if "use_detector" in config and config["use_detector"]:
        detector_param = config["detector"]

        detector_class, detector_config = utils.import_class_from_config(detector_param)
        detector = detector_class(detector_config)
    else:
        detector = None


    # init all the policy
    initial_policies = []
    for policy_name in config["initial_policies"]:
        policy_param = config["initial_policies"][policy_name]

        if policy_param['params'] is None:
            policy_param['params'] = dict()
        # add name to the parameter
        policy_param['params']["policy_name"] = policy_name
        policy_class, policy_config = utils.import_class_from_config(policy_param)
        policy = policy_class(policy_config, detector)
        initial_policies.append(policy)
    print(f"find {len(initial_policies)} initial policies")

    # init all objects
    objs_config = utils.config_from_yaml_file(task_config_file)
    objects_to_cluster = []
    for obj_name, obj_config in objs_config['objs'].items():
        obj_config_class = MeshObject.Config().update(obj_config)
        obj = MeshObject(obj_config_class, obj_name)
        objects_to_cluster.append(obj)
    print(f"find {len(objects_to_cluster)} objects to cluster")

    updated_config = PolicyCompressor.Config().update(config)
    compressor = PolicyCompressor(updated_config,
                                  initial_policies=initial_policies,
                                  output_file=output_file,
                                  run_name = run_name
                 )

    nmeshes = len(objects_to_cluster)
    for mesh_id, meshobj in enumerate(objects_to_cluster):
        print(f"============ {run_name} processing object {mesh_id}/{nmeshes} ============ ")
        compressor.add_object(meshobj)
        if (mesh_id + 1) %5 == 0:
            compressor.output_xml()
    compressor.output_xml()
    compressor.print_policy_summary()

if __name__=="__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    main()
