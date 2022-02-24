import xml.etree.ElementTree as ET
import os

mug_path = '/Users/gspat/Downloads/bowls'
save_path = '/Users/gspat/quant_codes/quantized_policies/Archive/collect_bowl_data_xmls'
if not os.path.exists(save_path):
    os.makedirs(save_path)

all_elems = os.listdir(mug_path)

filtered_elems = []
# for p in all_elems:
#     if "mug" in p:
#         filtered_elems.append(os.path.join(mug_path, p))

for p in all_elems:
    filtered_elems.append(os.path.join(mug_path, p))

print(len(filtered_elems))

for fe in filtered_elems:
    a_xml = fe
    print(a_xml)
    assert os.path.exists(a_xml)

    tree = ET.parse(a_xml)
    root = tree.getroot()

    mesh_elems = list()
    for m in root.iter('mesh'):
        mesh_elems.append(m)

    share_xml_path = '/Users/gspat/quant_codes/quantized_policies/Archive/share.xml'
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
    cnt = 0
    for mm in shared_root.iter('mesh'):
        print(mm)
        cnt += 1
    shared_tree.write(os.path.join(save_path, f'shared_bowl_{mesh_name}.xml'))

    ### ... now aadd the geom elements to the test_xml as well ... ###
    test_xml_path = '/Users/gspat/quant_codes/quantized_policies/Archive/test.xml'
    assert os.path.exists(test_xml_path)
    test_tree = ET.parse(test_xml_path)
    test_root = test_tree.getroot()
    children_of_root = test_root.getchildren()
    for child in children_of_root:
        if child.tag == 'include':
            child.attrib['file'] = f'shared_bowl_{mesh_name}.xml'
    worldbody_elem = children_of_root[3]
    worldbody_children = worldbody_elem.getchildren()
    for child in worldbody_children:
        if child.attrib['name'] == 'object0':
            object0_body = child

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

    print(mug_body)
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
        print(g)
        cnt += 1
    print(f'# elems copied: {cnt}')
    object0_body = test_root.getchildren()[3].getchildren()[3]
    for c in object0_body.getchildren():
        print(c.tag, c.attrib)
    test_tree.write(os.path.join(save_path, f'test_bowl_{mesh_name}.xml'))
