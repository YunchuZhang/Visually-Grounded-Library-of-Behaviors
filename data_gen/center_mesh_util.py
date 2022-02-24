import pcp_utils
import os
import yaml
import trimesh
import numpy as np
import xml.etree.ElementTree as ET

# add gym and baselines to the dir
gym_path = pcp_utils.utils.get_gym_dir()
baseline_path = pcp_utils.utils.get_baseline_dir()
os.sys.path.append(gym_path)
os.sys.path.append(baseline_path)

def maybe_check_obj_mean(mesh_path):
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    mean_vertex = mesh.vertices.mean(axis=0)
    return mean_vertex

def get_mesh_paths(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    asset_tag = None
    for root_child in iter(root):
        if root_child.tag == 'asset':
            asset_tag = root_child

    if asset_tag is None:
        raise ValueError('given xml does not contain asset element')

    mesh_paths = list()
    for asset_child in iter(asset_tag):
        if asset_child.tag == 'mesh':
            mesh_paths.append(asset_child.attrib['file'])
    return mesh_paths


def maybe_refine_mesh_paths(mesh_paths, source_mesh_dir):
    refined_mesh_paths = list()
    for m in mesh_paths:
        if os.path.exists(m):
            refined_mesh_paths.append(m)
        else:
            remove_str = '../meshes/'
            new_path = os.path.join(source_mesh_dir,
                    m[len(remove_str):])
            refined_mesh_paths.append(new_path)

    return refined_mesh_paths


def compute_correction_factor(mesh_paths, combined=False):
    meshes = [trimesh.load(m) for m in mesh_paths]
    if combined:
        combined_mesh = np.sum(meshes)
        
        # compute the mean
        mean_position = combined_mesh.vertices.mean(axis=0)
    else:
        # for each mesh I will compute the geometric center
        mean_position = [m.vertices.mean(axis=0) for m in meshes]
    return mean_position

def compute_correction_factor_bbox(mesh_paths):
    meshes = [trimesh.load(m) for m in mesh_paths]
    combined_mesh = np.sum(meshes)

    # compute the bounding box bounds
    bbox_bounds = combined_mesh.bounding_box.bounds
    # compute the center from the bounds
    bbox_center = bbox_bounds[0, :] + ((bbox_bounds[1, :] - bbox_bounds[0, :])/2.0)
    return bbox_center


def correct_meshes_and_save(correction_factor, mesh_paths):
    # load all the meshes
    meshes = [trimesh.load(m) for m in mesh_paths]
    if isinstance(correction_factor, list):
        for i, (mesh, corr_factor) in enumerate(zip(meshes, correction_factor)):
            mesh.vertices -= corr_factor
            mesh.export(mesh_paths[i])
    else: #It is an integer and I am using the combined mesh correction
        # from all meshes.vertices subtract the correction factor
        for mp, m in zip(mesh_paths, meshes):
            m.vertices -= correction_factor
            m.export(mp)

def check_meshes(mesh_paths, combined=False):
    meshes = [trimesh.load(m) for m in mesh_paths]
    eps = np.zeros(3,)
    if combined:
        combined_mesh = np.sum(meshes)
        mean_pos = combined_mesh.vertices.mean(axis=0)

        truth_val = np.allclose(mean_pos, eps)
    else: # for each mesh check that the mean is zero
        truths = [np.allclose(mesh.vertices.mean(axis=0), eps) for mesh in meshes]
        truth_val = all(truths)
    return truth_val

def check_meshes_bbox(mesh_paths):
    meshes = [trimesh.load(m) for m in mesh_paths]
    combined_mesh = np.sum(meshes)

    bbox_bounds = combined_mesh.bounding_box.bounds
    check = np.absolute(bbox_bounds[0,:]) - np.absolute(bbox_bounds[1,:])
    eps = 1e-8
    return np.all(np.absolute(check) < eps)

def center_from_xml_path(xmls_folder, correct_for, source_mesh_dir):
    """
        This uses the mean of the vertices to center the mesh, after
        this if you do, mesh.vertices.mean(axis=0), it will be (0,0,0)
    """
    for xml_folder in correct_for:
        xml_folder_path = os.path.join(xmls_folder, xml_folder)
        assert os.path.exists(xml_folder_path)
        all_xmls = os.listdir(xml_folder_path)
        for i, xml in enumerate(all_xmls):
            xml_path = os.path.join(xml_folder_path, xml)
            if xml_path.endswith('.xml'):
               mesh_paths = get_mesh_paths(xml_path)
               mesh_paths = maybe_refine_mesh_paths(mesh_paths, source_mesh_dir)
               # load all the meshes, combine them compute mean and return
               correction_factor = compute_correction_factor(mesh_paths, combined=False)
               # now that you have the correction factor correct all the meshes
               correct_meshes_and_save(correction_factor, mesh_paths)
               # all the meshes are corrected and saved finally check
               assert check_meshes(mesh_paths, combined=False), f"{xml_path} meshes don't have mean to be zero"

def center_from_xml_using_bbox(xmls_folder, correct_for, source_mesh_dir):
    """
        The meshes I have are arranged such that when I do mesh.vertices.mean(axis=0)
        the value would come out to be (0, 0, 0) but this is not required, I want to
        shift the mesh such that the center of the bounding box comes out to be (0,0,0)
    """
    for xml_folder in correct_for:
        xml_folder_path = os.path.join(xmls_folder, xml_folder)
        assert os.path.exists(xml_folder_path)
        all_xmls = os.listdir(xml_folder_path)
        for i, xml in enumerate(all_xmls):
            xml_path = os.path.join(xml_folder_path, xml)
            # the following check exists because there is a mysterious folder cups inside
            # cups which should not be taken into account while doing this...
            if xml_path.endswith('.xml'):
                mesh_paths = get_mesh_paths(xml_path)
                mesh_paths = maybe_refine_mesh_paths(mesh_paths, source_mesh_dir)
                # load all the meshes, combine them compute bbox and return center as
                # the correction factor to subtract from the vertices of all meshes.
                correction_factor = compute_correction_factor_bbox(mesh_paths)
                # subtract correction factor and save, now here mesh.vertices.mean()
                # would be zero no more.
                correct_meshes_and_save(correction_factor, mesh_paths)
                # all the meshes are corrected and saved, check them once.
                assert check_meshes_bbox(mesh_paths),\
                        f"{xml_path} does not have bbox center to be (0, 0, 0)"
               

def center_from_old_xmls_bus(xml_dir, source_mesh_dir):
    """
        Legacy code: should be able to safely remove in a few days
    """
    # Step1: collect all the xmls
    all_xmls = [x for x in os.listdir(xml_dir)]
    # Step2: filter out the above xmls so they only contain the shared part
    filtered_xmls = [x for x in all_xmls if 'shared_bus' in x]
    # Step3: check if you can get the mesh paths using the function defined
    for i, xml in enumerate(filtered_xmls):
        xml_path = os.path.join(xml_dir, xml)
        if xml_path.endswith('.xml'):
            mesh_paths = get_mesh_paths(xml_path)
            mesh_paths = maybe_refine_mesh_paths(mesh_paths, source_mesh_dir)
            ## subtract the correction factor
            correction_factor = compute_correction_factor_bbox(mesh_paths)
            correct_meshes_and_save(correction_factor, mesh_paths)
            assert check_meshes_bbox(mesh_paths),\
                    f"{xml_path} does not have bbox center to be (0,0,0)"


if __name__ == '__main__':
    source_mesh_dir = pcp_utils.utils.get_mesh_dir()
    
    source_xml_dir = pcp_utils.utils.get_object_xml_dir()
    correct_for_xmls = ['cups', 'bowls']
    
    center_from_xml_using_bbox(source_xml_dir, correct_for_xmls,
            source_mesh_dir)
