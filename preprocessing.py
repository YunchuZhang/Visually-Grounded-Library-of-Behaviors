# Includes utility functions required for data preprocessing
import numpy as np
import argparse
import os
import trimesh
import open3d as o3d
from pointcloud_generator import PointCloudGenerator

def get_bbox_corners(mesh, pcd=None, tgt_xyz=[0,0,0.46]):
    extents = mesh.bounding_box.extents
    corners = transform_boxes_to_corners(extents[0], extents[1], extents[2])
    rotated_corners = convert_box_to_ref_T_mujoco(corners)[:, :3]
    rotated_corners += tgt_xyz
    visualize_bboxes(rotated_corners, pcd)
    return rotated_corners, extents

def visualize_bboxes(corners, pcd):
    #import pdb; pdb.set_trace()
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    lines = np.asarray(lines, dtype=int)
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    if pcd is None:
        o3d.visualization.draw_geometries([line_set])
    else:
        pcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([line_set, pcd])

def convert_box_to_ref_T_mujoco(corners):
    # Corners are (8, 3) We need (8, 4) shape
    corners = np.hstack((corners, np.zeros((8, 1))))

    # Rotation matrix to rotate from shapenet to mujoco coordinate system
    rot_mat = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    rotated_corners = np.matmul(rot_mat, corners.T).T
    return rotated_corners

def transform_boxes_to_corners(lx, ly, lz):
    # returns corners, shaped [8 x 3]
    xs = np.stack([-lx/2., lx/2., -lx/2., lx/2., -lx/2., lx/2., -lx/2., lx/2.])
    ys = np.stack([-ly/2., -ly/2., -ly/2., -ly/2., ly/2., ly/2., ly/2., ly/2.])
    zs = np.stack([-lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2., lz/2.])

    corners = np.stack([xs, ys, zs], axis=1)
    return corners

def get_mesh_obj_stl(path_to_stl_folder):
    stl_files = os.listdir(path_to_stl_folder)
    for stl_file in stl_files:
        if "visual_convex_model_normalized" in stl_file:
            return os.path.join(path_to_stl_folder, stl_file)
    return None

def process_mesh(mesh_folder, translation, display=True):
    trans = translation
    mesh_obj_stl = get_mesh_obj_stl(mesh_folder)
    mesh = trimesh.load(mesh_obj_stl)
    convex_mesh = mesh.convex_hull
    # convex_mesh = convex_mesh.apply_scale(1.5)
    convex_mesh.vertices = convex_mesh.vertices - convex_mesh.centroid
    if display:
        gen = PointCloudGenerator()
        points = gen.sample_faces(convex_mesh.vertices, convex_mesh.faces)
        rot_mat = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        points = np.c_[points, np.ones(len(points))]
        points = np.matmul(rot_mat, points.T).T[:, :3]
        points += translation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        bbox_corners, extents = get_bbox_corners(convex_mesh, pcd, tgt_xyz=trans)
    else:
        bbox_corners = get_bbox_corners(convex_mesh, tgt_xyz=trans)
    return extents, bbox_corners


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_folder',
                        default=None,
                        type=str,
                        help='Path to stl file of a mesh.')
    parser.add_argument('--display',
                    default=True,
                    type=bool,
                    help='Display pointcloud of the mesh and the bounding box around it.')
    parser.add_argument('--translation',
                    nargs='+',
                    required=True,
                    type=float,
                    help='usage --translation <space separated_translation>')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    lengths, bbox_corners = process_mesh(args.mesh_folder, args.translation, args.display)