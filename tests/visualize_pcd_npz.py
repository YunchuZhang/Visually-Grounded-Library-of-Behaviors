
import click
import numpy as np
import pickle
import open3d as o3d

import pcp_utils


@click.command()
@click.argument("pcd_file")#config
@click.option("--mode")#config
def main(pcd_file, mode):

    if mode == "generated_grasps":
        pcds = np.load(pcd_file, allow_pickle=True).item()
    
        scene_pcd = pcp_utils.np_vis.get_pcd_object(pcds['pcd'], clip_radius=10.0)

        #'generated_scores': generated_scores,
        #import ipdb; ipdb.set_trace()
        #scene_pcd = pcp_utils.np_vis.get_pcd_object(pcds['obj_pcd'], clip_radius=10.0)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.zeros(3), size=0.8)
        things_to_print = [scene_pcd, frame]

        num_grasps = len(pcds['generated_scores'])
        threshold = 0 #0.95

        good_grasps_ids = [id_ for id_ in range(num_grasps) if pcds['generated_scores'][id_] > threshold]
        pcds['generated_grasps'] = [pcds['generated_grasps'][id_] for id_ in good_grasps_ids]
        num_grasps = len(pcds['generated_grasps'])

        for grasp_id in range(num_grasps):
            rot = pcds['generated_grasps'][grasp_id]
            grasp_linset = pcp_utils.np_vis.make_lineset_4x4matrix(rot)
            things_to_print += [grasp_linset]

        o3d.visualization.draw_geometries(things_to_print)
    else:

        if mode == "pcds_color":
            out = np.load(pcd_file, allow_pickle=True).item()
            pcds = out["pts"]
            color = out["color"]

        else:
            pcds = np.load(pcd_file)

    
        object_pcd = pcp_utils.np_vis.get_pcd_object(pcds, clip_radius=10.0)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.zeros(3), size=0.8)
    
        things_to_print = [object_pcd, frame]
        o3d.visualization.draw_geometries(things_to_print)



if __name__=="__main__":
	main()

