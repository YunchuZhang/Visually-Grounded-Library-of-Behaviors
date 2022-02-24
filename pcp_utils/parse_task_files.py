
import os
import trimesh
import xml.etree.ElementTree as ET
import pcp_utils
import numpy as np
import transformations

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# old generate integrated xml, better check again
#def generate_integrated_xml(env_xml_file, obj_xml_file, scale=1.0,
#                            mass=None, euler=None,
#                            add_bbox_indicator=False,
#                            randomize_color=False,
#                            prefix="",
#                            remove_site=False):

def generate_integrated_xml(env_xml_file, obj_xml_file, scale=1.0,
                            mass=None, euler=None,
                            target_size=None,
                            add_bbox_indicator=False,
                            randomize_color=False,
                            include_obj_info=False,
                            verbose=False,
                            prefix="",
                            obj_name=None,
                            remove_site=True):

    """
    This function takes an env xml file, and put an obj in the obj_xml_file
    to the env to create a integrated file describing the final env.

    inputs:
        env_xml_file: a xml file under quantize-gym/gym/envs/robotics/assets
        obj_xml_file: obsolute path for the obj xml
        scale: scale of the object
        add_bbox_indicator: [True or False] if True add a box object with free joint
            which does not take part in the collision
    output:
        final_xml_file: an integrated xml file (relative path) under quantize-gym/gym/envs/robotics/assets
    """
    # get the extent of the bounding box
    coords, combined_mesh = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(
            obj_xml_file, np.asarray([0, 0, 0]), scale=scale, euler=euler, return_combined_mesh=True)

    # center of the mesh
    center = combined_mesh.vertices.mean(axis=0)
    #print(f'center is {center}')
    centroid = combined_mesh.centroid

    # center_str, centroid_str
    center_str = ' '.join([str(c) for c in center])
    centroid_str = ' '.join([str(c) for c in centroid])

    obj_tree = ET.parse(obj_xml_file)
    obj_root = obj_tree.getroot()

    # find the mesh in the obj file
    mesh_elems = list()
    is_mesh_obj = False
    for m in obj_root.iter('mesh'):
        is_mesh_obj = True
        if 'scale' not in m.attrib:
            m.attrib['scale'] = "1 1 1"

        scale_from_xml = [float(s) for s in  m.attrib['scale'].split(" ")]
        scale_from_xml = [str(s * scale) for s in scale_from_xml]
        m.attrib['scale'] = " ".join(scale_from_xml)
        mesh_elems.append(m)

    # find the geom in the obj file
    obj_collision_body = None
    obj_body = obj_root.find('worldbody').find('body')
    #print(obj_body, obj_body.attrib['name'])
    #if obj_body.attrib['name'] == 'puck':
    for obj_body_part in  obj_body.findall('body'):
        if obj_body_part.attrib['name'] == 'collision':
            obj_collision_body = obj_body_part
            break

    if obj_collision_body is None:
        print(f"cannot find collision body in the object file")
        raise ValueError


    # load the env file
    gym_xml_path = os.path.join(pcp_utils.utils.get_gym_dir(), 'gym/envs/robotics/assets')
    full_env_xml_file = os.path.join(gym_xml_path, env_xml_file)
    if not os.path.exists(full_env_xml_file):
        print(f"cannot find the env file: {full_env_xml_file}")
    env_tree = ET.parse(full_env_xml_file)
    env_root = env_tree.getroot()

    #insert object asset (meshes) to the file after the include
    insert_id = 0
    new_asset = ET.Element('asset')
    for mesh in mesh_elems:
        mfp = mesh.attrib['file']
        mfp = pcp_utils.utils.maybe_refine_mesh_path(mfp)
        mesh.attrib['file'] = mfp
        new_asset.append(mesh)

    for elem_id, elem in enumerate(env_root.iter()):

        if elem.tag == "include":
            insert_id = elem_id
            break

    env_root.insert(insert_id - 1,  new_asset)# obj_root.find('asset'))

    # find object in the worldbody and replace it with geom from the obj xml
    worldbody_root = env_root.find('worldbody')
    # todo this part 
    object_to_find = 'object0'
    object_body_in_env = None
    for body in worldbody_root.iter('body'):
        if body.attrib['name'] ==  object_to_find:
            object_body_in_env = body

    if  object_body_in_env is None:
        print(f"cannot find {object_to_find} placeholder in the env file")
        raise ValueError

    # replace the object geom in the env
    for g in object_body_in_env.findall('geom'):
        object_body_in_env.remove(g)


    num_geoms = len(obj_collision_body.findall('geom'))
    for g in obj_collision_body.findall('geom'):

        if euler is not None:
            # rotate each geom
            if 'euler' not in g.attrib:
                current_euler = [0, 0, 0]
            else:
                # zyx
                current_euler = [float(a) for a in g.attrib['euler'].split(" ")][::-1]
            alpha, beta, gamma = current_euler
            dalpha, dbeta, dgamma = euler
            rot_mat = transformations.euler_matrix(alpha, beta, gamma)
            d_rot_mat = transformations.euler_matrix(dalpha, dbeta, dgamma)
            new_euler = transformations.euler_from_matrix(np.matmul(d_rot_mat, rot_mat), 'rxyz')
            euler_str = " ".join([str(a) for a in new_euler])

            g.attrib['euler'] = euler_str

        g.attrib["solref"] = "0.000002 1"
        g.attrib["solimp"] = "0.99 0.99 0.01"

        if "mass" not in g.attrib:
            g.attrib["mass"] = str(0.01/num_geoms)

        if mass is not None:
            g.attrib["mass"] = str(mass/num_geoms)
        # condim="4" mass="0.01"]

        if randomize_color:
            rgb = np.random.rand(3).tolist()
            rgb_str = " ".join([str(f) for f in rgb])
            rgb_str += " 1"
            g.attrib["rgba"] = rgb_str #"1 0 0 1"

        if not is_mesh_obj and scale is not None: # native geom object
            if 'size' not in g.attrib:
                g.attrib['size'] = "1 1 1"
            if 'pos' not in g.attrib:
                g.attrib['pos'] = "0 0 0"

            size_from_xml = [float(s) for s in  g.attrib['size'].split(" ")]
            size_from_xml = [str(s * scale) for s in size_from_xml]
            g.attrib['size'] = " ".join(size_from_xml)

            pos_from_xml = [float(s) for s in  g.attrib['pos'].split(" ")]
            pos_from_xml = [str(s * scale) for s in pos_from_xml]
            g.attrib['pos'] = " ".join(pos_from_xml)

        #g.attrib.pop("solimp")
        #g.attrib.pop("solref")

        object_body_in_env.append(g)

    num_site = len(obj_collision_body.findall('site'))
    if num_site > 0:
        for s in object_body_in_env.findall('site'):
            object_body_in_env.remove(s)
        for s in obj_collision_body.findall('site'):
            # here add the center string as the site
            object_body_in_env.append(s)

    if remove_site:
        # making them super small
        for body in worldbody_root.iter('body'):
            for s in body.findall('site'):
                s.attrib['rgba'] = "0 0 1 0"
        for site in worldbody_root.findall('site'):
            site.attrib['rgba'] = "0 0 1 0"


    # site_elem = object_body_in_env.findall('site')
    # site_elem[0].attrib['pos'] = center_str

    # add the indicator bbox object if necessary
    if add_bbox_indicator:
        bounds, center, extents = pcp_utils.np_vis.get_bbox_attribs(coords)
        # Mujoco requires half sizes of the box took me few hours to figure it out
        extents = extents / 2.0
        size = ' '.join([str(e) for e in extents])

        # create a body subelement in worldbody_element
        body_element = ET.SubElement(worldbody_root, 'body',
                name='bbox_indicator', pos='0 0 0')
        # add the geom to the body element
        ET.SubElement(body_element, 'geom', name="bbox_indicator_geom",
                type='box', contype='0', conaffinity='0',
                size=size, rgba='0 0 1 0.3')
        # finally add a free joint to the indicator
        ET.SubElement(body_element, 'joint', type='free',
                name='bbox_indicator_joint')

    indent(env_root)

    if prefix == "":
        final_xml_file =  f'fetch/generated_env.xml'
    else:

        final_xml_file =  f'fetch/{prefix}_generated_env.xml'
    env_tree.write(os.path.join(gym_xml_path,  final_xml_file))
    return final_xml_file
