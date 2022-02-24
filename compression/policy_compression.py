
import pcp_utils
import sys
import os

gym_path = pcp_utils.utils.get_gym_dir()
baseline_path = pcp_utils.utils.get_baseline_dir()

sys.path.append(gym_path)
sys.path.append(baseline_path)


##import os
#os.system(f"export PYTHONPATH={gym_path}:{baseline_path}:$PYTHONPATH")

##import ipdb; ipdb.set_trace()


import argparse
import pcp_utils.load_ddpg as load_ddpg
from pcp_utils.rollouts import simple_rollouts

import tqdm
import numpy as np


##### Imports related to environment #############
import gym
#from compression.config.meshes import MESHES

MESHES = [
	'159e56c18906830278d8f8c02c47cde0',
	'159e56c18906830278d8f8c02c47cde0',
	# #'6661c0b9b9b8450c4ee002d643e7b29e',
	# 'b9004dcda66abf95b99d2a3bbaea842a',
	#'c39fb75015184c2a0c7f097b1a1f7a5',
	#'ec846432f3ebedf0a6f32a8797e3b9e9',
	#'f99e19b8c4a729353deb88581ea8417a'
]

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_data_path',
						default='/Users/sfish0101/Documents/2020/Spring/quantized_policies/trained_models_fetch/fetch_cups',
						type=str,
						help='Path to some initial expert data collected.')
	parser.add_argument('--max-path-length', '-l', type=int, default=50)
	parser.add_argument('--num-rollouts', '-n', type=int, default=10)
	args = parser.parse_args()

	return args

def load_policy(expert_data_path, mesh_id):
	# Load the policy corresponding to the source mesh
	load_path='{}/{}/{}'.format(args.expert_data_path, mesh_id,'/models/save55')
	params_path='{}/{}/{}'.format(args.expert_data_path, mesh_id,'logs')

	expert_policy = load_ddpg.load_policy(load_path, params_path)
	return expert_policy

def get_mesh_xml_path(mesh_id, task='pick_and_place_cup_'):
	return '{}{}.xml'.format(task, mesh_id)

class PolicyCompressor:
	def __init__(self, expert_data_path,
						num_rollouts=100,
						max_path_length=50,
						accept_threshold=0.9,
						num_threads=5):

		self.expert_data_path = expert_data_path
		self.num_rollouts = num_rollouts
		self.max_path_length = max_path_length
		self.policy_bank = {}
		self.clusters = {}
		self.cluster_lookup = {}

		self.cluster_acc = {}

		self.num_clusters = 0
		self.accept_threshold = accept_threshold
		self.num_threads = num_threads

		# TODO: load prestroed policies

	# Access to a minimal policy bank, also has information about which meshes to run on which policy
	# Takes in a new object and determines if it can be merged with an existing policy or can spawn a new policy
	# input: new object that needs to be classified
	# mesh should be mesh_id like 159e56c18906830278d8f8c02c47cde0, or b9004dcda66abf95b99d2a3bbaea842a which are ShapeNet ids
	def compress_policies_online(self, mesh):

		if len(self.policy_bank.keys()) == 0:
			self.policy_bank[mesh] = load_policy(self.expert_data_path, mesh)
			self.num_clusters += 1
			self.clusters["c" + str(self.num_clusters)] = [mesh]
			self.cluster_lookup[mesh] = "c" + str(self.num_clusters)
			self.cluster_acc["c" + str(self.num_clusters)] = [(mesh, 1)]
		else:
			success_rates = []
			mesh_vals = []

			print("Starting rollouts for src {} tgt {} ".format(mesh, mesh))
			
			xml_path = get_mesh_xml_path(mesh)
			#initialize environments with this mesh
			print("Initializing environment with ", xml_path)
			env = gym.make('FetchPickAndPlace-v1', model_xml_path=xml_path)

			expert_policy = load_policy(self.expert_data_path, mesh)
			# perform rollouts to gather stats
			stats = simple_rollouts(env,
						expert_policy,
						args.num_rollouts,
						args.max_path_length)

			base_success_rate = stats['success_rate']

			# Compute accuracy for each of the clusters
			for cid, meshes in self.clusters.items():
				# load policy of the first mesh (parent mesh) in an existing cluster
				expert_policy = load_policy(self.expert_data_path, meshes[0])
				print("Checking performance of {} on policy for {}".format(mesh, meshes[0]))
				# perform rollouts to gather stats
				stats = simple_rollouts(env,
						expert_policy,
						args.num_rollouts,
						args.max_path_length)

				success_rate = stats['success_rate']
				
				mesh_vals.append(meshes[0])
				success_rates.append(success_rate)

			env.close()
			max_success_rate = np.max(success_rates)

			print("Max Success Rate ", max_success_rate, " Base success rate ", base_success_rate)
			if max_success_rate >= self.accept_threshold: #* base_success_rate:
				max_idx = np.argmax(success_rates)
				best_mesh = mesh_vals[max_idx]
				cval = self.cluster_lookup[best_mesh]
				self.clusters[cval].append(mesh)
				self.cluster_acc[cval].append((mesh, success_rates[max_idx]))
			else:
				# Form a new cluster
				self.policy_bank[mesh] = load_policy(self.expert_data_path, mesh)
				self.num_clusters += 1
				self.clusters["c" + str(self.num_clusters)] = [mesh]
				self.cluster_lookup[mesh] = "c" + str(self.num_clusters)
				self.cluster_acc["c" + str(self.num_clusters)] = [(mesh, base_success_rate)]


	def compress_policies(self):
		for mesh in tqdm.tqdm(MESHES):
			self.compress_policies_online(mesh)
		print("Compressed Meshes")
		print(self.clusters)
		print("Percentage compression for threshold {} : {} ".format(self.accept_threshold, self.num_clusters/len(MESHES)))
		print("Cluster Accuracies")
		print(self.cluster_acc)

def main(args):

	compressor = PolicyCompressor(args.expert_data_path, num_rollouts=args.num_rollouts)
	compressor.compress_policies()

if __name__=="__main__":
	args = parse_args()
	main(args)
