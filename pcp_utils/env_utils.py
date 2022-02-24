import numpy as np
import trimesh


import pcp_utils


def get_bbox_properties(env, obj_info):

    obj_xml = obj_info.obj_xml_file
    obj_xpos = env.env.sim.data.get_body_xpos('object0')
    obj_xmat = env.env.sim.data.get_body_xmat('object0')
    obj_xquat = env.env.sim.data.get_body_xquat('object0')
    scale = obj_info.scale
    coords, combined_mesh = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(
        obj_info.obj_xml_file, obj_xpos, obj_xmat, scale=scale, euler=obj_info.euler, return_combined_mesh=True)

    # now get the properties
    bounds, center, extents = pcp_utils.np_vis.get_bbox_attribs(coords)

    # check here if bounding box is fine for the object
    # I will draw the box using my computed values
    transform = np.eye(4)
    transform[:3, 3] = center
    bounding_box_outline = trimesh.primitives.Box(
        transform=transform, extents=extents
    )
    bounding_box_outline.visual.face_colors = [0, 0, 255, 100]

    # just to make sure that the bounding box is tight here
    assert np.allclose(bounding_box_outline.bounds, combined_mesh.bounds)

    # # plot the box and mesh
    # scene = trimesh.Scene()
    # scene.add_geometry([combined_mesh, bounding_box_outline])
    # scene.show()
    ### ... checking of bounding box for every step ends ... ###
    bbox_xpos = center.copy()

    return bounds, center, extents, obj_xquat, bbox_xpos



def reset_everything_on_table(env, mesh_obj_info, max_run=100):
    _, center, _, obj_xquat, bbox_xpos = get_bbox_properties(env, mesh_obj_info)
    center_old = center
    for i in range(max_run):
        obsDataNew, reward, done, info = env.step(np.zeros(8))
        _, center, _, obj_xquat, bbox_xpos = get_bbox_properties(env, mesh_obj_info)
        if np.linalg.norm(center_old - center) < 0.000001:
            return
        center_old = center
    return