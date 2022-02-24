import numpy as np
import trimesh 
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mesh_path',
						default=None,
						type=str,
						help='Path to stl file of a mesh.')
	args = parser.parse_args()
	return args

class PointCloudGenerator:

	def load_mesh(self, path):
		mesh = trimesh.load(path)
		return mesh

	# Function to sample pointcloud given vertices and faces of a mesh
	# Copied from: https://github.com/gsp-27/mujoco_hand_exps/blob/659b2b30042680fbafea8a764a7267c8add9d8c8/trajectory_env/dataset_generator.py#L262-L314
	def sample_faces(self, vertices, faces, n_samples=5000):
		"""
		Samples point cloud on the surface of the model defined as vectices and
		faces. This function uses vectorized operations so fast at the cost of some
		memory.
		Parameters:
		vertices  - n x 3 matrix
		faces     - n x 3 matrix
		n_samples - positive integer
		Return:
		vertices - point cloud
		Reference :
		[1] Barycentric coordinate system
		\begin{align}
			P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
		\end{align}
		"""
		actual_n_samples = n_samples
		vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
							vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
		face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
		face_areas = face_areas / np.sum(face_areas)

		# Sample exactly n_samples. First, oversample points and remove redundant
		# Error fix by Yangyan (yangyan.lee@gmail.com) 2017-Aug-7
		n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
		floor_num = np.sum(n_samples_per_face) - n_samples
		if floor_num > 0:
			indices = np.where(n_samples_per_face > 0)[0]
		floor_indices = np.random.choice(indices, floor_num, replace=True)
		n_samples_per_face[floor_indices] -= 1

		n_samples = np.sum(n_samples_per_face)

		# Create a vector that contains the face indices
		sample_face_idx = np.zeros((n_samples, ), dtype=int)
		acc = 0
		for face_idx, _n_sample in enumerate(n_samples_per_face):
			sample_face_idx[acc: acc + _n_sample] = face_idx
			acc += _n_sample

		r = np.random.rand(n_samples, 2);
		A = vertices[faces[sample_face_idx, 0], :]
		B = vertices[faces[sample_face_idx, 1], :]
		C = vertices[faces[sample_face_idx, 2], :]
		P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
			np.sqrt(r[:,0:1]) * r[:,1:] * C
		return P

	def generate_pointcloud(self, mesh_path):
		mesh = self.load_mesh(mesh_path)
		vertices, faces = mesh.vertices, mesh.faces
		pointcloud = self.sample_faces(vertices, faces)
		return pointcloud
		
def main(args):
	gen = PointCloudGenerator()
	pointcloud = gen.generate_pointcloud(args.mesh_path)
	cloud = trimesh.points.PointCloud(pointcloud)
	sc = trimesh.Scene([cloud])
	sc.show()

if __name__=="__main__":
	args = parse_args()
	main(args)
