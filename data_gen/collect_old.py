import mujoco_py
import os
import PIL.Image as Image
from dm_control import mjcf
# from dm_control import viewer
import matplotlib
import matplotlib.pyplot as plt
import utils
import xml.etree.ElementTree as ET
import numpy as np
from attrdict import AttrDict

#os.sys.path.append('/Users/gspat/quant_codes/quantized_policies')
from preprocessing import process_mesh
from bounding_box import bounding_box as bb
import cv2
from itertools import combinations
import open3d as o3d


import pcp_utils

def modify_and_change_xml(xml_string, stl_files_path, xml_save_path):
    with open(xml_save_path, 'w') as f:
        f.write(xml_string)

    tree = ET.parse(xml_save_path)
    root = tree.getroot()

    ## need to write stuff again changing the paths of model files
    assets = root.findall('./asset')
    cnt = 0
    for m in assets[0].getchildren():
        if m.tag == 'mesh':
            m.attrib['file'] = os.path.join(stl_files_path[cnt])
            m.attrib.pop('class')
            cnt += 1
    tree.write(xml_save_path)

def split_intrinsics(intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    return fx, fy, cx, cy

def naive_filter_points(points, threshold_low=0.38, threshold_high=0.5):
    cond1 = points[:, 2] > threshold_low
    cond2 = points[:, 2] < threshold_high
    cond = cond1 & cond2
    filtered_points = points[cond, :]

    # visualize the points here once
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    o3d.visualization.draw_geometries([pcd])	
    print(f'# of points belonging to the object are {len(filtered_points)}')
    return filtered_points

def draw_boxes_on_image_py(rgb, corners_pix, scores, tids,info_text=None, boxes=None, thickness=1,text=False):
    # all inputs are numpy tensors
    # rgb is H x W x 3
    # corners_pix is N x 8 x 2, in xy order
    # scores is N
    # tids is N
    # boxes is N x 9 < this is only here to print some rotation info
    # pix_T_cam is 4 x 4
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    corners_pix = np.floor(corners_pix)
    corners_pix = corners_pix.astype(int)
    H, W, C = rgb.shape
    assert(C==3)
    N, D, E = corners_pix.shape
    assert(D==8)
    assert(E==2)

    if boxes is not None:
        rx = boxes[:,6]
        ry = boxes[:,7]
        rz = boxes[:,8]
    else:
        rx = 0
        ry = 0
        rz = 0

    color_map = matplotlib.cm.get_cmap('tab20')
    color_map = color_map.colors

    # draw
    for ind, corners in enumerate(corners_pix):
        # corners is 8 x 2
        if not np.isclose(scores[ind], 0.0):
            # print 'score = %.2f' % scores[ind]
            color_id = tids[ind] % 20
            color = color_map[2]
            color_text = color_map[2]

            # st()

            color = np.array(color)*255.0
            # print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])
            if info_text is not None:
                text_to_put = info_text[ind]
                cv2.putText(rgb,
                    text_to_put, 
                    (np.min(corners[:,0]), np.min(corners[:,1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, # font size
                    color_text,
                    2) # font weight

                # NOTE: I know my ordering
            lines = [[0, 1], [0, 2], [1, 2], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
            cx = np.clip(corners[:, 0], 0, W-1)
            cy = np.clip(corners[:, 1], 0, H-1)
            corners_n = np.c_[cx, cy]
            for l in lines:
                cv2.line(rgb, tuple(corners_n[l[0]]), tuple(corners_n[l[1]]), color, thickness, cv2.LINE_AA)
            # for c in corners:
            # 	# rgb[pt1[0], pt1[1], :] = 255
            # 	# rgb[pt2[0], pt2[1], :] = 255
            # 	# rgb[np.clip(int(c[0]), 0, W), int(c[1]), :] = 255

            # 	c0 = np.clip(int(c[0]), 0,  W-1)
            # 	c1 = np.clip(int(c[1]), 0,  H-1)
            # 	rgb[c1, c0, :] = 255

            # we want to distinguish between in-plane edges and out-of-plane ones
            # so let's recall how the corners are ordered:
            # xs = np.array([-1/2., 1/2., -1/2., 1/2., -1/2., 1/2., -1/2., 1/2.])
            # ys = np.array([-1/2., -1/2., 1/2., 1/2., -1/2., -1/2., 1/2., 1/2.])
            # zs = np.array([-1/2., -1/2., -1/2., -1/2., 1/2., 1/2., 1/2., 1/2.])
            # xs = np.reshape(xs, [8, 1])
            # ys = np.reshape(ys, [8, 1])
            # zs = np.reshape(zs, [8, 1])
            # offsets = np.concatenate([xs, ys, zs], axis=1)

            # corner_inds = list(range(8))
            # combos = list(combinations(corner_inds, 2))

            # for combo in combos:
            # 	pt1 = offsets[combo[0]]
            # 	pt2 = offsets[combo[1]]
            # 	# draw this if it is an in-plane edge
            # 	eqs = pt1==pt2
            # 	if np.sum(eqs)==2:
            # 		i, j = combo
            # 		pt1 = (corners[i, 0], corners[i, 1])
            # 		pt2 = (corners[j, 0], corners[j, 1])
            # 		retval, pt1, pt2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
            # 		if retval:
            # 			cv2.line(rgb, pt1, pt2, color, thickness, cv2.LINE_AA)

                    # rgb[pt1[0], pt1[1], :] = 255
                    # rgb[pt2[0], pt2[1], :] = 255
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # utils_basic.print_stats_py('rgb_uint8', rgb)
    # imageio.imwrite('boxes_rgb.png', rgb)
    return rgb


def compute_bounding_box_from_pts(pts):
    """
        pts: N, 3 array of object points
        returns: bbox --> an o3d.geometry.AxisAlignedBox object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    bbox = pcd.get_axis_aligned_bounding_box()

    # for sanity checking see that the bounding box is proper
    o3d.visualization.draw_geometries([pcd, bbox])
    return bbox


def clip_outside_radius(pts, radius=0.8, object_center=None):
    """
        pts: (N, 3) containing a lot of extra points which are far away
        radius: points lying beyond the sphere of this radius are dropped.
        object_center: if specified represents the center of the object
                       and is a 3d np.ndarray

        returns: inlier pts (M, 3) np.ndarray
    """
    N, _ = list(pts.shape)
    distance_from_origin = np.linalg.norm(pts, axis=1)
    assert distance_from_origin.shape[0] == N

    filter_idxs = distance_from_origin < radius
    inliers = pts[filter_idxs, :]
    return inliers


def segment_plane(points):
    """
        points: (N, 3) np.ndarray containing a plane
        returns: (M, 3) np.ndarray containing only object points
    """
    # now filter out some of the points, using the radius
    inlier_pts = clip_outside_radius(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inlier_pts)

    # now use the segment plane functionality
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
        ransac_n=3,
        num_iterations=250)

    # GET THE OUTLIER POINTS, THESE ARE POINTS WHICH ARE NOT IN THE PLANE
    outlier_pcd = pcd.select_down_sample(inliers, invert=True)

    # these also includes far away points which do not belong to the cup
    outlier_pts = np.asarray(outlier_pcd.points)
    cup_points = clip_outside_radius(outlier_pts)

    # visualize the cup_points for sanity checking
    cup_pcd = o3d.geometry.PointCloud()
    cup_pcd.points = o3d.utility.Vector3dVector(cup_points)

    o3d.visualization.draw_geometries([cup_pcd])
    return cup_points


def make_lineset(points):
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
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
        )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def compute_bounding_box_python(pts):
    max_bound = pts.max(axis=0)
    min_bound = pts.min(axis=0)

    front_face_bottom_left = np.asarray([min_bound[0], min_bound[1], min_bound[2]])
    front_face_bottom_right = np.asarray([max_bound[0], min_bound[1], min_bound[2]])
    front_face_top_left = np.asarray([min_bound[0], min_bound[1], max_bound[2]])
    front_face_top_right = np.asarray([max_bound[0], min_bound[1], max_bound[2]])
    back_face_bottom_left = np.asarray([min_bound[0], max_bound[1], min_bound[2]])
    back_face_bottom_right = np.asarray([max_bound[0], max_bound[1], min_bound[2]])
    back_face_top_left = np.asarray([min_bound[0], max_bound[1], max_bound[2]])
    back_face_top_right = np.asarray([max_bound[0], max_bound[1], max_bound[2]])

    # now draw it once for sanity checking
    points = np.stack([
        front_face_bottom_left,
        front_face_bottom_right,
        front_face_top_left,
        front_face_top_right,
        back_face_bottom_left,
        back_face_bottom_right,
        back_face_top_left,
        back_face_top_right
        ])	
    return points


def get_box_corners_2d(bbox_3d, int_mat):
    """
        bbox_3d : (N, 3) np.ndarray
        int_mat : intrinsic matrix
    """
    fx, fy, cx, cy = split_intrinsics(int_mat)
    X, Y, Z = bbox_3d[:, 0], bbox_3d[:, 1], bbox_3d[:, 2]
    x_pix = ((fx * X) / Z) + cx
    y_pix = ((fy * Y) / Z) + cy
    box_2d = np.stack((x_pix, y_pix), axis=1)
    return box_2d


def main(xml_path, mesh_name=None, mesh_dir_path=None):
    current_folder = os.getcwd()
    xml_path = os.path.join(current_folder, xml_path)
    color_img_dir = os.path.join(current_folder, 'images')
    if not os.path.exists(color_img_dir):
        os.makedirs(color_img_dir)
    instance_storage_path = os.path.join(current_folder, 'instance')
    if not os.path.exists(instance_storage_path):
        os.makedirs(instance_storage_path)

    ## mujoco_py loading, do you really need this, actually this is
    ## required to render the environment, this is just for sanity checking
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    for _ in range(5000):
    	sim.step()
    while True:
    	viewer.render()
    ## ... End of mujoco py setup ... ##

    ## ... The code for collecting data actually begins here ... ##
    visual_mjcf = mjcf.from_path(xml_path)
    lookat_pos = visual_mjcf.worldbody.body['object0'].pos
    center_of_movement = lookat_pos
    camera_positions, camera_quats = utils.generate_new_cameras(0.3,
        center_of_movement,
        lookat_pos,
        height=0.5,
        jitter_z=True,
        jitter_amount=0.08)

    ep_imgs = list()
    ep_depths = list()
    ep_intrinsics = list()
    ep_extrinsics = list()
    img_height = 128
    img_width = 128
    fov_y = 45
    save_image = False
    recon_scene = True
    mesh_dir = os.path.join("/Users/sfish0101/Documents/2020/Spring/quantized_policies/data_gen/meshes/03797390",
        mesh_name)
    mesh_dir = os.path.join(mesh_dir, "models/stl_models")

    # based on the mesh name also get the bounding box for the mesh and transform it to every
    # camera coordinate system
    physics = mjcf.Physics.from_mjcf_model(visual_mjcf)
    physics.forward()
    for _ in range(5000):
        physics.step()

    # now get the object position
    object_xpos = physics.data.xpos[physics.model.name2id('object0', 'body')]

    # now I need to call the file processing.py to get the bounding box corners
    # lengths, bbox_corners_origin= process_mesh(mesh_dir, object_xpos)
    # bbox_corners_origin *= 1.5

    # now for each camera position and orientation get the image, depth, intrinsics and extrinsics
    for i, (pos, quat) in enumerate(zip(camera_positions, camera_quats)):
        print(f'generating for {i}/{len(camera_positions)} ...')
        visual_mjcf.worldbody.add('camera', name=f'vis_cam:{i}', pos=pos, quat=quat, fovy=fov_y)
        physics = mjcf.Physics.from_mjcf_model(visual_mjcf)

        # okay here the physics is initialized but since the object is placed at a location
        # not respecting the actual size fo the object I should simulate phyics for sometime
        physics.forward()
        for _ in range(5000):
            physics.step()

        img = physics.render(img_height, img_width, camera_id=f'vis_cam:{i}')
        depth = physics.render(img_height, img_width, camera_id=f'vis_cam:{i}', depth=True)

        assert img.shape[0] == img_height, "color img height is wrong"
        assert img.shape[1] == img_width, "color img width is wrong"
        assert depth.shape[0] == img_height, "depth img height is wrong"
        assert depth.shape[1] == img_width, "depth img width is wrong"
        if save_image:
            fig, ax = plt.subplots(2, sharex=True, sharey=True)
            ax[0].imshow(img)
            ax[1].imshow(depth)
            fig.savefig(f"{color_img_dir}/img_{i}.png")
            plt.close(fig=fig)

        # get the intrinsics and extrinsics and form the dict and store the data
        intrinsics = utils.dm_get_intrinsics(fov_y, img_width, img_height)
        extrinsics = utils.dm_get_extrinsics(physics, physics.model.name2id(
            f'vis_cam:{i}', 'camera'
            ))

        ############# BOUNDING BOX DRAWING START #####################################################################
        # now that you have the extrinsics which is from camera to origin, inverse of this will
        # give me from origin to camera coordinate system
        # cam_T_origin = np.linalg.inv(extrinsics)
        # bbox_t = np.c_[bbox_corners_origin, np.ones(len(bbox_corners_origin))]
        # bbox_corners_cam = np.dot(cam_T_origin, bbox_t.T).T
        # # so the bounding box is now in camera coordinate system, project it to image coordinate system
        # fx, fy, cx, cy = split_intrinsics(intrinsics)
        # X, Y, Z = bbox_corners_cam[:, 0], bbox_corners_cam[:, 1], bbox_corners_cam[:, 2]
        # x_pix = ((fx*X) / Z) + cx
        # y_pix = ((fy*Y) / Z) + cy

        # x_pix = 128 - x_pix
        # y_pix = 128 - y_pix

        # now I will use Adam's draw_box function
        # rgb = draw_boxes_on_image_py(img, corners_pix=np.c_[x_pix, y_pix][None], scores=np.ones(1,), tids=np.ones(1,))
        # plt.imshow(rgb)
        # plt.show()
        ######## BOUNDING BOX DRAWING AND REGISTRATION COMPLETE #######################################################

        ep_imgs.append(img)
        ep_depths.append(depth)
        ep_intrinsics.append(intrinsics)
        ep_extrinsics.append(extrinsics)

    visual_mjcf.worldbody.add('camera', name='ref_cam', pos = [0,-0.3,0.5], zaxis = [0,-1,0], fovy=fov_y)
    physics = mjcf.Physics.from_mjcf_model(visual_mjcf)
    physics.forward()
    # let it settle
    for _ in range(5000):
        physics.step()
    img = physics.render(img_height, img_width, camera_id='ref_cam')
    depth = physics.render(img_height, img_width, camera_id='ref_cam', depth=True)
    intrinsics = utils.dm_get_intrinsics(45, img_width, img_height)
    extrinsics = utils.dm_get_extrinsics(physics, physics.model.name2id(
        'ref_cam', 'camera'
        ))

    if save_image:
        fig, ax = plt.subplots(2, sharex=True, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(depth)
        fig.savefig(f'{color_img_dir}/img_ref.png')
    ep_imgs.append(img)
    ep_depths.append(depth)
    ep_intrinsics.append(intrinsics)
    ep_extrinsics.append(extrinsics)

    # next recreate the scene !!
    if recon_scene:
        # Here I can also return the points in the world and the points in ref_cam
        # this would remove the need for scaling rotation, translation etc
        recon_imgs, world_xyzs, camR_xyzs = utils.recreate_scene(ep_depths,
            ep_intrinsics,
            ep_extrinsics,
            camR_T_origin = np.linalg.inv(ep_extrinsics[-1]),
            clip_radius=5.0)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.zeros(3), size=0.8)
        # now here visualize the points first and then compute the bounding box using plane subtraction
        # collect all the points together
        world_xyzs_collated = np.concatenate(world_xyzs,axis=0)
        cup_points = segment_plane(world_xyzs_collated)
        # next compute the bounding box using these points
        bbox_points = compute_bounding_box_python(cup_points)
        bbox_lineset = make_lineset(bbox_points)

        # rotate the cup points, rotate the bbox points and see if they match
        camR_T_origin = np.linalg.inv(ep_extrinsics[-1])
        h_cup = np.c_[cup_points, np.ones(len(cup_points))]
        r_cup = np.dot(camR_T_origin, h_cup.T).T[:, :3]
        # rotate the box
        h_bbox_points = np.c_[bbox_points, np.ones(len(bbox_points))]
        r_bbox_points = np.dot(camR_T_origin, h_bbox_points.T).T[:, :3]
        print(r_bbox_points)

        # make the rotated pcd
        r_cup_pcd = o3d.geometry.PointCloud()
        r_cup_pcd.points = o3d.utility.Vector3dVector(r_cup)
        r_bbox_lineset = make_lineset(r_bbox_points)
        # make the pcd and visualize
        cup_pcd = o3d.geometry.PointCloud()
        cup_pcd.points = o3d.utility.Vector3dVector(cup_points)
        # o3d.visualization.draw_geometries([cup_pcd, bbox_lineset, frame])
        # o3d.visualization.draw_geometries([r_cup_pcd, r_bbox_lineset, frame])

        # project the bounding box in the reference camera here
        ref_box_2d = get_box_corners_2d(r_bbox_points, int_mat=ep_intrinsics[-1])
        ref_box_2d = ref_box_2d[None]
        ref_rgb = draw_boxes_on_image_py(ep_imgs[-1], ref_box_2d, scores=np.ones(1), tids=np.ones(1))
        # plt.imshow(ref_rgb)
        # plt.show()

        for e, ext_mat in enumerate(ep_extrinsics):
            # also project the box onto the image of each camera
            temp_points = np.c_[bbox_points, np.ones(len(bbox_points))]
            box_in_camX = np.dot(np.linalg.inv(ext_mat), temp_points.T).T[:, :3]
            box_2d_camX = get_box_corners_2d(box_in_camX, int_mat=ep_intrinsics[e])
            box_2d_camX = box_2d_camX[None]
            box_img = draw_boxes_on_image_py(ep_imgs[e], box_2d_camX, scores=np.ones(1), tids=np.ones(1))
            # plt.imshow(box_img)
            # plt.show()

        for j, im in enumerate(recon_imgs):
            im = np.asarray(im)
            im = (im * 255.).astype(np.uint8)
            r_im = Image.fromarray(im)
            r_im.save(os.path.join(instance_storage_path, f'visual_recon_img_{j}.png'))

    # create a dictionary to save the data and ship it !!
    save_dict = AttrDict()
    save_dict.rgb_camXs = np.stack(ep_imgs)
    save_dict.depth_camXs = np.stack(ep_depths)
    save_dict.intrinsics = np.stack(ep_intrinsics)
    save_dict.extrinsics = np.stack(ep_extrinsics)
    save_dict.camR_T_origin = np.linalg.inv(ep_extrinsics[-1])
    save_dict.bbox_in_ref_cam = r_bbox_points
    rgb_camRs = np.reshape(ep_imgs[-1], [1, img_height, img_width, 3])
    rgb_camRs = np.tile(rgb_camRs, [len(ep_imgs), 1, 1, 1])
    save_dict.rgb_camRs = rgb_camRs
    # everything should be len(51)
    for k in save_dict.keys():
        if k == 'camR_T_origin' or k == 'bbox_in_ref_cam':
            continue
        assert len(save_dict[k]) == 51, "Data specific length is not right"
    vis_save_path = os.path.join(current_folder, f"visual_data_{mname}.npy")
    np.save(vis_save_path, save_dict)
    print('---- done ----')


def get_mesh_name(f):
    splits = f.split('/')
    xml_path = splits[-1]
    more_splits = xml_path.split('_')
    mesh_name = more_splits[-1][:-4]
    return mesh_name


if __name__ == '__main__':

    root_dir = pcp_utils.utils.get_root_dir()
    dir_path = os.path.join(root_dir, "quantized_policies/data_gen/collect_bowl_data_xmls")
    #'/Users/gspat/quant_codes/quantized_policies/Archive/collect_bowl_data_xmls'
    assert os.path.exists(dir_path)
    all_files = os.listdir(dir_path)
    filtered_file_paths = []
    for f in all_files:
        # so that I do not select the texture directory or the share file
        if 'shared' not in f and 'texture' not in f:
            filtered_file_paths.append(os.path.join(dir_path, f))
    for f in filtered_file_paths:
        mname = get_mesh_name(f)
        print(f'working for {mname}')
        main(f, mesh_name=mname)
