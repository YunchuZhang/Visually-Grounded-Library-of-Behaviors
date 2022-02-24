import os
import argparse
import sys
import numpy as np
import open3d as o3d

import pcp_utils

#graspnet_dir = '/Users/sfish0101/Documents/2020/Spring/pointnet_6d_grasp/6dof-graspnet'
graspnet_path = pcp_utils.utils.get_6dof_graspnet_dir()
sys.path.append(graspnet_path)
import tf_utils

def make_pcd(pts):
    assert pts.shape[1] == 3, "give me 3d points"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def make_frame():
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

def split_grasps(grasp):
    pos = grasp[:3, 3]
    orn = grasp[:3, :3]
    return pos, orn

def make_lineset(pts):
    pts = pts.copy()
    lines = [[0,1], [1,2], [1,5], [2,3], [5,6]]
    lineset = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(pts),
        lines = o3d.utility.Vector2iVector(lines)
    )
    return lineset

def main(args):


    if args.dir_path is not None:
        if not os.path.exists(args.dir_path):
            raise FileNotFoundError
    
        npy_files = os.listdir(args.dir_path)
        npy_files = [os.path.join(args.dir_path, npy_file) for npy_file in npy_files]
    else:
        npy_files = [args.file_path]


    mujoco_T_adam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    origin_T_camR_xpos = np.array([1.3, 0.75, 0.4], np.float32) + np.array([0, 0, 0], np.float)

    origin_T_adam = np.zeros((4,4), dtype=np.float32)
    origin_T_adam[:3, :3] = mujoco_T_adam
    origin_T_adam[:3, 3] = origin_T_camR_xpos
    origin_T_adam[3,3] = 1
    origin_T_adam = origin_T_adam
    adam_T_origin = np.linalg.inv(origin_T_adam)

    for i, f in enumerate(npy_files):
        if args.dir_path is not None:
            fp = os.path.join(args.dir_path, f)
        fp = f
        print("filename:", fp)
        data = np.load(fp, allow_pickle=True).item()
        gripper_pc = np.squeeze(tf_utils.get_control_point_tensor(1, False), 0)

        gripper_pc[2, 2] = 0.059
        gripper_pc[3, 2] = 0.059
        mid_point = 0.5*(gripper_pc[2, :] + gripper_pc[3, :])

        # modified grasps
        modified_gripper_pc = []
        modified_gripper_pc.append(np.zeros((3,), np.float32))
        modified_gripper_pc.append(mid_point)
        modified_gripper_pc.append(gripper_pc[2])
        modified_gripper_pc.append(gripper_pc[4])
        modified_gripper_pc.append(gripper_pc[2])
        modified_gripper_pc.append(gripper_pc[3])
        modified_gripper_pc.append(gripper_pc[5])


        gripper_pc_ori = np.asarray(modified_gripper_pc)

        # object pcd visualization with respect to gripper
        obj_pts = data['obj_pcd']


        #num_pts = obj_pts.shape[0]
        #obj_pts_addone = np.concatenate([obj_pts, np.ones((num_pts, 1))], axis=1)
        #obj_pts = np.matmul(origin_T_adam, obj_pts_addone.T).T[:, :3]
        obj_pcd = make_pcd(obj_pts)
        
        # filter:
        #        num_grasps = len(data['generated_scores'])
        #        # under the table
        #
        #        keep_id = []
        #        for grasp_id in range(num_grasps):
        #            a_grasp = data['generated_grasps'][grasp_id]
        #            grasp_pos, grasp_orn = split_grasps(a_grasp)
        #            gripper_pc = np.matmul(gripper_pc_ori, grasp_orn.T)
        #            gripper_pc += grasp_pos[None]
        #
        #
        #            if np.max(gripper_pc, 0)[1] < 0: #going into table
        #                keep_id.append(grasp_id)
        #
        #
        #        #keep_id = [grasp_id for grasp_id in range(num_grasps) if data['generated_grasps'][grasp_id][1, 3] < 0]
        #
        #        data['generated_grasps'] = [data['generated_grasps'][id_] for id_ in keep_id]
        #        data['generated_scores'] = [data['generated_scores'][id_] for id_ in keep_id]
        #
        #        print("#original_grasps:", num_grasps, "#grasps after filtering", len(data["generated_grasps"]))

        things_to_print = [make_frame(), obj_pcd]

        top_k = min(30, len(data['generated_scores']))

        grasp_ids = np.argsort(data['generated_scores'])[-top_k:][::-1]

        for rank_id in range(0, top_k):
            grasp_id = grasp_ids[rank_id]
            # now select one of the grasps and visualize the point
            a_grasp = data['generated_grasps'][grasp_id]

            grasp_pos, grasp_orn = split_grasps(a_grasp)

            #            grasp_pos_addones = np.ones((4, 1), np.float32)
            #            grasp_pos_addones[:3, 0] = grasp_pos
            #            grasp_pos = np.matmul(origin_T_adam, grasp_pos_addones)[:3, 0]
            #
            #            grasp_orn_addones = np.eye(4)
            #            grasp_orn_addones[:3, :3] = grasp_orn
            #            grasp_orn = np.matmul(origin_T_adam, grasp_orn_addones)
            #            grasp_orn = grasp_orn[:3, :3]


            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.translate(grasp_pos)
    
            # rotate the gripper points
            gripper_pc = np.matmul(gripper_pc_ori, grasp_orn.T)
            gripper_pc += grasp_pos[None]
            lineset = make_lineset(gripper_pc)
            things_to_print += [make_pcd(gripper_pc), lineset, sphere]
        o3d.visualization.draw_geometries(things_to_print)
        #[make_pcd(gripper_pc), make_frame(), obj_pcd, sphere, lineset])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=None)
    parser.add_argument('--file_path', type=str, default=None)

    args = parser.parse_args()
    main(args)
