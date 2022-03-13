import numpy as np
import pickle
import open3d as o3d
import os
from tqdm import tqdm

import pcp_utils
def visualize(images, bbox_points_from_mesh=None):
    depths = images['depth_camXs']
    pix_T_cams = images['pix_T_cams']
    origin_T_camXs = images['camR_T_camXs']
    _, xyz_camRs, _ = pcp_utils.np_vis.unproject_depth(depths,
        pix_T_cams,
        origin_T_camXs,
        camR_T_origin = None, #np.linalg.inv(self.origin_T_adam),
        clip_radius=5.0,
        do_vis=False)

    # 3 is a bit off
    all_xyz_camR = np.concatenate(xyz_camRs[0:1], axis=0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.zeros(3), size=0.8)
    object_pcd = pcp_utils.np_vis.get_pcd_object(all_xyz_camR, clip_radius=3.0)
    things_to_print = [object_pcd, frame]
    if bbox_points_from_mesh is not None:
        bbox_lineset_from_mesh = pcp_utils.np_vis.make_lineset(bbox_points_from_mesh)
        things_to_print.append(bbox_lineset_from_mesh)
    # transform object xpos and xmat to the adam coordinate (x right, y downs)
    #bbox_lineset_from_mesh_adam = pcp_utils.np_vis.make_lineset(bbox_points_from_mesh_adam)
    o3d.visualization.draw_geometries(things_to_print) #, bbox_lineset_from_mesh_adam])

import ipdb
st=ipdb.set_trace

FOLDER_NAME = 'data_0624_r1'


#files = [f"/projects/katefgroup/quantized_policies/real_data/object{i}.npy" for i in range(56)]
files = [f"/home/sirdome/workspace/data_cluttered/{FOLDER_NAME}/object{i}.npy" for i in range(5)]
# files = [f"/home/sirdome/workspace/iccv_data_yunchu/object{i}.npy" for i in range(34, 42)]
# file = "/home/sirdome/workspace/iccv_data_yunchu/object41.npy"
for file in tqdm(files):
    data = np.load(file,allow_pickle=True).item()


    #data['rgb_camXs']


    #rgbs = data['rgb_camXs']
    #import imageio
    #for i in range(5):
    #    imageio.imwrite(f"data/y{i}.png", rgbs[i])

    import cv2
    import imageio
    from scipy import interpolate
    data["pix_T_cams"] = data["intrinsic"]
    data["origin_T_camXs"] = data["extrinsic"]

    # adjust intrinsics
    # crop rgb and depth
    rgb_camXs = data['rgb_camXs']
    ncams, H, W, _ = rgb_camXs.shape

    crop_range_x = list(range(W))
    crop_range_y = list(range(H))
    xx, yy = np.meshgrid(crop_range_x, crop_range_y)


    crop_size = 128
    resize_scale = crop_size/480

    resize_rgb_camXs = []
    new_intrinsics = []
    new_depths = []
    for cam_id in range(ncams):
    	rgb = rgb_camXs[cam_id]
    	depth = data["depth_camXs"][cam_id]
    	imageio.imwrite(f"tmp/rgb_{cam_id}.png", rgb)


    	fr = interpolate.interp2d(crop_range_x, crop_range_y, rgb[:,:,0], kind="linear")
    	fg = interpolate.interp2d(crop_range_x, crop_range_y, rgb[:,:,1], kind="linear")
    	fb = interpolate.interp2d(crop_range_x, crop_range_y, rgb[:,:,2], kind="linear")
    	fd = interpolate.interp2d(crop_range_x, crop_range_y, depth, kind="linear")

    	intrinsic = data["pix_T_cams"][cam_id].reshape((3,3))

    	xnew =  np.array(list(range(crop_size)))
    	ynew = np.array(list(range(crop_size)))

    	c_w = intrinsic[0, 2]
    	c_h = intrinsic[1, 2]
    	f_w = intrinsic[0, 0]
    	f_h = intrinsic[1 ,1]

    	new_intrinsic = np.array([[f_w * resize_scale,                  0, crop_size*0.5],
    		                       [                 0, f_h * resize_scale, crop_size*0.5],
    		                       [                 0,                  0, 1]])
    	new_intrinsics.append(new_intrinsic)
    	xnew = (1/resize_scale) * (xnew - (crop_size-1)/2) + c_w
    	ynew = (1/resize_scale) * (ynew - (crop_size-1)/2) + c_h

    	#xx_new, yy_new = np.meshgrid(xnew, ynew)
    	new_image_r = fr(xnew, ynew)
    	new_image_g = fg(xnew, ynew)
    	new_image_b = fb(xnew, ynew)
    	new_image = np.stack([new_image_r, new_image_g, new_image_b], axis=2).astype(np.uint8)

    	new_image_d = fd(xnew, ynew)
    	new_depths.append(new_image_d)


    	#center_cropped_rgb = rgb[:, 80:561, :]
    	#resize_rgb = cv2.resize(center_cropped_rgb, dsize=(128, 128))
    	#imageio.imwrite(f"tmp/resize2_{cam_id}.png", new_image)

    	resize_rgb_camXs.append(new_image)

    data['rgb_camXs'] = np.stack(resize_rgb_camXs, axis=0)
    data["pix_T_cams"] = np.stack(new_intrinsics, axis=0)
    data["depth_camXs"] = np.stack(new_depths, axis=0)/1000.0



    resize_scale = 128/480

    #data["pix_T_cams"] = 


    del data["intrinsic"]
    del data["extrinsic"]


    adam_T_yun = np.array([[ 0,  1,  0,   0],
                  [ 0,  0, -1,   0],
                  [-1,  0,  0, 0.4],
                  [ 0,  0,  0,   1]])


    adam_T_camXs = []
    ncams = data["origin_T_camXs"].shape[0]
    for cam_id in range(ncams):

        adam_T_camX = np.matmul(adam_T_yun, data["origin_T_camXs"][cam_id])
        adam_T_camXs.append(adam_T_camX)

    adam_T_camXs = np.stack(adam_T_camXs)
    data["camR_T_camXs"] = adam_T_camXs
    del data["origin_T_camXs"]


    #data[]
    bounds = np.array(data["bbox"]).reshape(2, 3)
    x_center = bounds[0,0]
    x_len = bounds[1,0]
    y_center = bounds[0,1]
    y_len = bounds[1,1]
    z_center = bounds[0,2]
    z_len = bounds[1,2]

    # add more tolerance
    x_len = x_len * 1.5
    y_len = y_len * 1.5 
    z_len = z_len * 1.1


    x_min = x_center - x_len * 0.5
    x_max = x_center + x_len * 0.5
    # x_min = 0.34

    y_min = y_center - y_len * 0.5
    y_max = y_center + y_len * 0.5

    # y_min = -0.08
    # y_max = 0.01

    z_min = z_center - z_len * 0.5
    z_max = z_center + z_len * 0.5

    # z_max = 0.08
    # print(x_min, x_max, y_min, y_max, z_min, z_max)
    bbox_points_from_mesh = np.array([[x_min, y_min, z_min],
              [x_max, y_min, z_min],
              [x_min, y_min, z_max],
              [x_max, y_min, z_max],
              [x_min, y_max, z_min],
              [x_max, y_max, z_min],
              [x_min, y_max, z_max],
              [x_max, y_max, z_max],
              ])

    data["bbox_camR"] = bbox_points_from_mesh
    del data["bbox"]
    # print(data.keys())

    if "success_claee" in data:
        data["success_rates_over_class"] = data["success_claee"]
        del data["success_claee"]
    else:
        data["success_rates_over_class"] = data["success_class"]
        del data["success_class"]
    bbox_points_from_mesh_adam = np.matmul(adam_T_yun, pcp_utils.geom.pts_addone(bbox_points_from_mesh).T).T[:,:3]

    # visualize(data, bbox_points_from_mesh_adam)
    # st()
    save_dir = f"/home/sirdome/workspace/quantize_training/data/plate/{FOLDER_NAME}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file.split("/")[-1].replace("object", "preprocessed_object")), data)