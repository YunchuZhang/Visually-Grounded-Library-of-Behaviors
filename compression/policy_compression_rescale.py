import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import utils.load_ddpg
import tqdm
from utils.rollouts import simple_rollouts
import numpy as np
from utils.utils import rescale_mesh_in_env
import json

##### Imports related to environment #############
import gym
from config.meshes import MESHES

scales = np.arange(0.3, 2, 0.1)
z_rotation_angles = np.arange(0, 2*np.pi, np.pi/6)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_data_path',
						default='/Users/apokle/Documents/quantized_policies/trained_models_fetch/fetch_cup',
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

	expert_policy = utils.load_ddpg.load_policy(load_path, params_path)
	return expert_policy

def get_mesh_xml_path(mesh_id, task='pick_and_place_cup_'):
	return '{}{}.xml'.format(task, mesh_id)

class PolicyCompressor:
	def __init__(self, expert_data_path,
						num_rollouts=2,
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
		self.all_accuracies = {}
		self.best_mesh_rot_angle = {}
		self.policy_success_rates = {}

	# Access to a minimal policy bank, also has information about which meshes to run on which policy
	# Takes in a new object and determines if it can be merged with an existing policy or can spawn a new policy
	# input: new object that needs to be classified
	# mesh should be mesh_id like 159e56c18906830278d8f8c02c47cde0, or b9004dcda66abf95b99d2a3bbaea842a which are ShapeNet ids
	def compress_policies_online(self, mesh, scale=1.2): # 1.2 is the default scale in the environment
		if (mesh, scale) in self.policy_bank.keys():
			return

		if len(self.policy_bank.keys()) == 0:
			# Stores only the best policy for an object
			# Note that objects of same mesh but different scales are considered as different objects

			self.policy_bank[(mesh, scale)] = load_policy(self.expert_data_path, mesh)
			self.num_clusters += 1
			self.clusters["c" + str(self.num_clusters)] = [(mesh, scale)]
			self.cluster_lookup[(mesh, scale)] = "c" + str(self.num_clusters)

			rescale_mesh_in_env(mesh, scale=scale)
			xml_path = get_mesh_xml_path(mesh)
			#initialize environments with this mesh
			print("Initializing environment with ", xml_path)
			env = gym.make('FetchPickAndPlace-v1', model_xml_path=xml_path)
			#env.render()

			expert_policy = load_policy(self.expert_data_path, mesh)
			# perform rollouts to gather stats
			stats = simple_rollouts(
						env=env,
						policy=expert_policy,
						num_rollouts=args.num_rollouts,
						path_length=args.max_path_length,
						#rescale_obs=True,
						)			

			self.cluster_acc["c" + str(self.num_clusters)] = [(mesh, stats['success_rate'])]
		else:

			print("Starting rollouts for src {} tgt {}".format(mesh, mesh))
			
			rescale_mesh_in_env(mesh, scale=1.2)
			xml_path = get_mesh_xml_path(mesh)
			#initialize environments with this mesh
			print("Initializing environment with ", xml_path)
			env = gym.make('FetchPickAndPlace-v1', model_xml_path=xml_path)
			#env.render()

			expert_policy = load_policy(self.expert_data_path, mesh)
			# perform rollouts to gather stats
			stats = simple_rollouts(
						env=env,
						policy=expert_policy,
						num_rollouts=args.num_rollouts,
						path_length=args.max_path_length,
						#rescale_obs=True,
						)
			success_rates = []
			mesh_vals = []

			base_success_rate = stats['success_rate']

			for rot_angle in z_rotation_angles:

				rot_success_rates = []
				rot_mesh_vals = []
				# Compute accuracy for each of the clusters
				for cid, meshes in self.clusters.items():
					print("Initializing environment with ", xml_path)
					env = gym.make('FetchPickAndPlace-v1', model_xml_path=xml_path, z_rotation_angle=rot_angle)

					# load policy of the first mesh (parent mesh) in an existing cluster
					#### TODO: Ideally this should load the best policy
					print(self.clusters.items())
					expert_policy = load_policy(self.expert_data_path, meshes[0][0])
					print("Checking performance of {} on policy for {} for rotation_angle {}".format(mesh, meshes[0], rot_angle))
					
					# perform rollouts to gather stats
					stats = simple_rollouts(
						env=env,
						policy=expert_policy,
						scale=scale,
						num_rollouts=args.num_rollouts,
						path_length=args.max_path_length
						)

					success_rate = stats['success_rate']
					
					rot_mesh_vals.append(meshes[0])
					rot_success_rates.append(success_rate)

					self.all_accuracies["{} {} {}".format(mesh, scale, rot_angle)] = success_rate
					
					env.close()

				# For a given rotation angle, gind the best success rate
				success_rates.append(np.max(rot_success_rates))
				# Find the rotation angle corresponding to the best success rate
				mesh_vals.append(rot_mesh_vals[np.argmax(rot_success_rates)])

			max_success_rate = np.max(success_rates)
			best_rotation = z_rotation_angles[np.argmax(success_rates)]

			self.policy_success_rates["{} {}".format(mesh, scale)] = (max_success_rate, best_rotation)
			
			print("Max Success Rate ", max_success_rate, " Base success rate ", base_success_rate)
			if max_success_rate >= self.accept_threshold: #* base_success_rate:
				max_idx = np.argmax(success_rates)
				best_mesh = mesh_vals[max_idx]
				cval = self.cluster_lookup[best_mesh]
				self.clusters[cval].append((mesh, scale))
				self.cluster_acc[cval].append((mesh, scale, success_rates[max_idx]))
			else:
				# Form a new cluster
				self.policy_bank[(mesh, scale)] = load_policy(self.expert_data_path, mesh)
				self.num_clusters += 1
				self.clusters["c" + str(self.num_clusters)] = [(mesh, scale)]
				self.cluster_lookup[(mesh, scale)] = "c" + str(self.num_clusters)
				self.cluster_acc["c" + str(self.num_clusters)] = [(mesh, base_success_rate)]

			print(self.clusters)
			print(self.cluster_acc)

	def init_policy_bank(self):
		for mesh in MESHES:
			self.compress_policies_online(mesh, scale=1.2)  #default scale used to train all objects in environment

	def compress_policies(self):
		self.init_policy_bank()
		for mesh in tqdm.tqdm(MESHES):
			for scale in scales:
				self.compress_policies_online(mesh, scale=scale)

			f1 = "all_accuracies_scaled_{}.json".format(mesh)
			f2 = "best_accuracies_scaled_{}.json".format(mesh)
			with open(f1, 'w') as outfile:
				json.dump(self.all_accuracies, outfile)
			outfile.close()

			with open(f2, 'w') as outfile:
				json.dump(self.policy_success_rates, outfile)
			outfile.close()

		print(self.policy_success_rates)

		print("Compressed Meshes")
		print(self.clusters)
		f3 = "clusters.json"
		with open(f3, 'w') as outfile:
			json.dump(self.clusters, outfile)
		outfile.close()

		print("Percentage compression for threshold {} : {} ".format(self.accept_threshold, self.num_clusters/len(MESHES)))
		print("Cluster Accuracies")
		print(self.cluster_acc)
		f4 = "clusters.json"
		with open(f4, 'w') as outfile:
			json.dump(self.cluster_acc, outfile)
		outfile.close()

def main(args):
	compressor = PolicyCompressor(args.expert_data_path, num_rollouts=args.num_rollouts)
	compressor.compress_policies()

if __name__=="__main__":
	args = parse_args()
	main(args)
