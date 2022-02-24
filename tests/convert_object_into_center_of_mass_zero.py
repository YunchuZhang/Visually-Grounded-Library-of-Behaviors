import trimesh
import os
import numpy as np
import xml.etree.ElementTree as ET

import pcp_utils

xml_file_root = "/Users/sfish0101/Documents/2020/Spring/quan_meshes/xmls/robotsuite_objects"
output_xml_file_root = "/Users/sfish0101/Documents/2020/Spring/quan_meshes/xmls/robotsuite_objects_centered"

xml_file_names = ["hammer8.xml"] #"nail1.xml", "nail2.xml", "nail3.xml", "nail4.xml"] #, "spray.xml"]

#""bottle2.xml", "bottle3.xml", "bottle4.xml", "camera.xml", "camera2.xml",  "hammer.xml", "hammer2.xml", "hammer3.xml",  "hammer3.xml",  "hammer4.xml",
#                   "hammer5.xml",  "hammer6.xml", "hammer7.xml", "hammer8.xml", "nail1.xml", "nail2.xml", "nail3.xml", "nail4.xml", "nail5.xml", "nail6.xml"] 

# "/Users/sfish0101/Documents/2020/Spring/quan_meshes/xmls/cups/159e56c18906830278d8f8c02c47cde0.xml"
#xml_file = "/Users/sfish0101/Documents/2020/Spring/quan_meshes/xmls/robotsuite_objects_centered/bottle.xml"
mesh_dir = pcp_utils.utils.get_mesh_dir()


for xml_file_name in xml_file_names:
    xml_file = os.path.join(xml_file_root, xml_file_name)

    asset_mesh, obj_geoms = pcp_utils.np_vis.get_meshes_from_xml(xml_file)
    meshes = []
    is_native_geom = False
    # get the combined meshes
    for geom_name, geom in obj_geoms.items():
        if geom['type'] == "mesh":
            mesh_name = geom_name
            file_name, mesh_scale = asset_mesh[mesh_name]
            mesh = trimesh.load(os.path.join(mesh_dir, file_name))
            mesh_scaled = mesh.apply_scale(mesh_scale[0])
            # if type is mesh, the size comment is ignored
            # http://www.mujoco.org/book/XMLreference.html#geom
            #geom_scale = geom['size'][0]
            #mesh_scaled_geom = mesh_scaled_xml.apply_scale(geom_scale)
            mesh_scaled = mesh_scaled.apply_translation(geom['pos'])
            meshes.append(mesh_scaled)
        else:
            is_native_geom  = True
            mesh_name = geom_name
            geom_type = geom['type']
            size = geom['size']
            if geom_type == "capsule":
                mesh_scaled_xml = trimesh.creation.capsule(radius=size[0], height=size[1]*2)
                # trimesh is putting center of capsule on the bottom center hemishpere of the capsule
                # so try to push the whole object down
                mesh_scaled_xml.apply_translation([0,0, -size[1]])
            elif geom_type == "cylinder":
                mesh_scaled_xml = trimesh.creation.cylinder(radius=size[0], height=size[1]*2)
            elif geom_type == "box":
                mesh_scaled_xml = trimesh.creation.box([s*2 for s in size])
            elif geom_type == "sphere":
                mesh_scaled_xml = trimesh.creation.uv_sphere(radius=size[0])
            else:
                print("such geom type is not supported. Please add code here")
                raise ValueError
            if "xmat" in geom:
                mesh_scaled_xml = mesh_scaled_xml.apply_transform(geom['xmat'])
            mesh_scaled = mesh_scaled_xml.apply_translation(geom['pos'])
            #mesh_scaled_xml.show()
            meshes.append(mesh_scaled)

    combined_mesh = np.sum(meshes)
    tmp  = combined_mesh.vertices.mean(axis=0)
    com  = combined_mesh.center_mass

    #import ipdb; ipdb.set_trace()

    # do the shift
    output_xml_file = os.path.join(output_xml_file_root, xml_file_name)

    #"/Users/sfish0101/Documents/2020/Spring/quan_meshes/xmls/cups_centered/159e56c18906830278d8f8c02c47cde0.xml"

    obj_tree = ET.parse(xml_file)
    obj_root = obj_tree.getroot()
    
    obj_collision_body = None
    obj_body = obj_root.find('worldbody').find('body')
    #print(obj_body, obj_body.attrib['name'])
    #if obj_body.attrib['name'] == 'puck':
    for obj_body_part in  obj_body.findall('body'):
        if obj_body_part.attrib['name'] == 'collision':
            obj_collision_body = obj_body_part
            break
    
    for g in obj_collision_body.findall('geom'):
        shifted_pos = np.array([float(s) for s in g.attrib['pos'].split(" ")]) - com
        g.attrib['pos'] = " ".join([str(t) for t in shifted_pos])
    
    obj_tree.write(output_xml_file)
    print("make file", output_xml_file)