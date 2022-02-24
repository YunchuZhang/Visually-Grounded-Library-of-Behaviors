#!/usr/bin/env python
# coding: utf-8

# In[1]:


import trimesh
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
import os.path as osp


# In[3]:

import numpy as np
import matplotlib.pyplot as plt

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

shapenet_path = '/home/gsp/Downloads/trimmed_shapenet/03797390'
results_path = '/home/gsp/Codes/pytorch_disco/touch_nearest_neighbour_results_on_val_data.npy'


# ## Format of Results
#
# Results is a ```np.ndarray``` each entry of this array is a ```dict``` containing, **object_name**, **true_location** and the **nearest neighbours** locations.
#
# ## Test 1:
# I claim that each of these points lie on the object itself so their distance to the closest point on the mesh should be zero.

# In[4]:


results = np.load(results_path, allow_pickle=True)
print('Length of results : {}'.format(len(results)))
save_dir = "touch_points_visualization_on_val_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# In[5]:

print('plotting the count')
categories = list()
for scene_item in results:
    # it is a dictionary
    name = scene_item['object']
    exists_flag = False
    for c in categories:
        if name in c:
            exists_flag = True
            break
    
    if not exists_flag:
        categories.append(f'scene_{name}')
    
from IPython import embed; embed()


# I first do it for just one entry and then write a full fledged general thing
for scene_num in range(len(results)):
    consideration = results[scene_num]
    print(consideration.keys())


    # In[6]:


    # step 1 load the object
    object_mesh_path = osp.join(shapenet_path, consideration['object_name'][0]) + '/models' + '/model_normalized.obj'
    print(object_mesh_path)


    # In[7]:


    mesh = trimesh.load(object_mesh_path)
    # mesh.show()


    # In[8]:


    # Step 2 rotate the mesh
    mesh = mesh.apply_transform(trimesh.transformations.euler_matrix(*np.deg2rad([90, 0, 0])))


    # In[9]:


    # mesh.show()


    # In[19]:

    if len(consideration['neighbour_locations_on_same_object']) == 0:
        continue
    closest, distance, t_id = mesh.nearest.on_surface(consideration['neighbour_locations_on_same_object'])
    closest1, distance1, t_id1 = mesh.nearest.on_surface([consideration['true_location']])
    print(distance)
    print(distance1)
    # In[21]:

    # now I need to find how to color the mesh's specific triangles
    # mesh.unmerge_vertices()
    # mesh.visual.face_colors = np.tile([[10, 20, 255, 255]], [len(mesh.faces), 1])
    # mesh.visual.face_colors[t_id] = [255, 255, 255, 255]
    # true_pointcloud = trimesh.points.PointCloud([consideration['true_location']])
    # true_pointcloud.colors = trimesh.visual.random_color()
    # predicted_pointcloud = trimesh.points.PointCloud(consideration['neighbour_locations_on_same_object'])

    # instead of making pointcloud I will make a sphere with constant radius at each of these locations
    true_points = trimesh.primitives.Sphere(radius=0.02, center=consideration['true_location'])
    true_points.visual.face_colors = [10, 20, 100, 200]

    # also create spheres for the nearest neighbours prediction
    predicted_points = list()
    for pt in consideration['neighbour_locations_on_same_object']:
        temp = trimesh.primitives.Sphere(radius=0.02, center=pt)
        if np.equal(pt, consideration['true_location']).all():
            # color me green
            temp.visual.face_colors = [20, 100, 10, 200]
        else:
            temp.visual.face_colors = [100, 20, 10, 200]
        predicted_points.append(temp)
    # from IPython import embed; embed()
    scene = trimesh.Scene([mesh, true_points, predicted_points])
    scene.show()

    # # offline render it baby and you are done.
    # filename = f'{save_dir}/scene_render_{scene_num}.png'
    # try:
    #     png = scene.save_image(resolution=[1920, 1080], visible=True)
    #     with open(filename, 'wb') as f:
    #         f.write(png)
    #         f.close()
    # except BaseException as E:
    #     print('unable to save image', str(E))

