import xml.etree.ElementTree as ET
import os
import sys
cur_path = os.getcwd()
sys.path.append(cur_path)
import pcp_utils

gym_dir = pcp_utils.utils.get_gym_dir()

# mug_path = '/Users/zyc/Downloads/cups'
bowls_path = os.path.join(gym_dir, "gym/envs/robotics/assets/fetch/bowls")
cups_path = os.path.join(gym_dir, "gym/envs/robotics/assets/fetch/cups")

root_dir = pcp_utils.utils.get_root_dir()
save_path = os.path.join(root_dir, "data_gen")

share_xml_path = os.path.join(root_dir, "data_gen/share.xml")
test_xml_path = os.path.join(root_dir, "data_gen/test.xml")

import ipdb; ipdb.set_trace()

assert os.path.exists(bowls_path)
assert os.path.exists(cups_path)
all_files = {}
all_paths = {}
bowl_files = os.listdir(bowls_path)
cup_files = os.listdir(cups_path)
# save a dic to search for different class obj's xml folder
all_files['bowl_files'] = bowl_files
all_files['cup_files'] = cup_files
all_paths['bowl_files'] = bowls_path
all_paths['cup_files'] = cups_path



def generate_share_and_test(xml_name):
    a_xml = xml_name + '.xml'
    for key,value in all_files.items():
        if a_xml in all_files[key]:
            cluster_name = key
            break
    print("---------------------------------------------")
    print("generating",cluster_name)
    print("generating",a_xml)
    a_xml = os.path.join(all_paths[cluster_name], a_xml)

    assert os.path.exists(a_xml)

    tree = ET.parse(a_xml)
    root = tree.getroot()

    mesh_elems = list()
    for m in root.iter('mesh'):
        mesh_elems.append(m)

    assert os.path.exists(share_xml_path)
    shared_tree = ET.parse(share_xml_path)
    shared_root = shared_tree.getroot()
    assets = shared_root.getchildren()
    for mm in assets[0].findall('mesh'):
        assets[0].remove(mm)
    # now need to append the elements of mug xml to shared xml
    for m in mesh_elems:
        # change the scale attrib and append
        m.attrib['scale'] = '1.2 1.2 1.2'
        assets[0].append(m)
    splits = mesh_elems[0].attrib['file'].split('/')
    mesh_name = splits[-4]
    print(f"mesh_name: {mesh_name}")

    shared_tree.write(os.path.join(save_path, 'save_shared.xml'))

    ### ... now aadd the geom elements to the test_xml as well ... ###
    assert os.path.exists(test_xml_path)
    test_tree = ET.parse(test_xml_path)
    test_root = test_tree.getroot()
    children_of_root = test_root.getchildren()
    # compiler_root = children_of_root[0]
    # mesh_path = os.path.join(gym_dir, "gym/envs/robotics/assets/stls/fetch")
    # texture_path = os.path.join(gym_dir, "gym/envs/robotics/assets")
    # compiler_root.attrib['meshdir'] = mesh_path
    # compiler_root.attrib['texturedir'] = texture_path
    for child in children_of_root:
        if child.tag == 'include':
            child.attrib['file'] = 'save_shared.xml'
            break
    worldbody_elem = children_of_root[3]
    worldbody_children = worldbody_elem.getchildren()
    # for child in worldbody_children:
    #     if child.attrib['name'] == 'object0':
    #         object0_body = child
    assert worldbody_children[3].attrib['name'] == 'object0'
    for g in worldbody_children[3].findall('geom'):
        worldbody_elem.getchildren()[3].remove(g)

    for g in worldbody_children[3].iter('geom'):
        print(g)

    if len(root.getchildren()) > 2:
        mug_worldbody = root.getchildren()[2]
        mug_body = mug_worldbody.getchildren()[1].getchildren()
    elif len(root.getchildren()) == 2:
        mug_worldbody = root.getchildren()[1]
        mug_body = mug_worldbody.getchildren()[0].getchildren()


    for m in mug_body:
        if m.attrib['name'] == 'collision':
            collision_body = m

    geoms_to_copy = list()
    for g in collision_body.iter('geom'):
        geoms_to_copy.append(g)

    print(len(geoms_to_copy))

    for g in geoms_to_copy:
        worldbody_children[3].append(g)
    cnt = 0
    for g in worldbody_children[3].iter('geom'):
        # print(g)
        cnt += 1
    print(f'# elems copied: {cnt}')
    print("----------------------------------------------")
    # object0_body = test_root.getchildren()[3].getchildren()[3]
    # for c in object0_body.getchildren():
    #     print(c.tag, c.attrib)
    test_tree.write(os.path.join(save_path, 'save_test.xml'))
if __name__ == '__main__':
    all_elems = os.listdir(bowls_path)

    print(len(all_elems))
    for xml_file in all_elems:
        # import ipdb;ipdb.set_trace()
        generate_share_and_test(xml_file[:-4])