import numpy as np
import math
from scipy.linalg import inv, sqrtm
import transformations



def sym(w):
    return w.dot(inv(sqrtm(w.T.dot(w))))


def render_images(env, camera_img_height, camera_img_width, camera_fov, camera_positions, camera_quats, camera_name="ext_camera_0"):
    """
    go through all the cameras and get rgbd
    """
    n_cams = len(camera_positions)

    rgbs = []
    depths = []
    pix_T_cams = []
    origin_T_camXs = []
    for cam_id in range(n_cams):
        # need to reset everytime you want to take the picture: the camera has mass and it will fall during execution
        env.set_camera(camera_positions[cam_id, :], camera_quats[cam_id, :], camera_name= camera_name)
        rgb, depth = env.render_from_camera(camera_img_height, camera_img_width, camera_name=camera_name)

        # need to convert depth to real numbers
        pix_T_camX = get_intrinsics(camera_fov, camera_img_width, camera_img_height)
        origin_T_camX = gymenv_get_extrinsics(env, camera_name)

        rgbs.append(rgb)
        depths.append(depth)
        pix_T_cams.append(pix_T_camX)
        origin_T_camXs.append(origin_T_camX)

    images = dict()
    images['rgb_camXs'] = np.stack(rgbs, axis=0)
    images['depth_camXs'] = np.stack(depths, axis=0)
    images['pix_T_cams'] = np.stack(pix_T_cams, axis=0)
    images['origin_T_camXs'] = np.stack(origin_T_camXs, axis=0)

    return images


def render_images_from_config(env, config):
    """
    go through all the cameras and get rgbd
    """
    camera_img_width = config['img_width']
    camera_img_height = config['img_height']
    camera_fov = config['fov_y']
    camera_positions = config['pos']
    camera_quats = config['quat']
    camera_name = config['camera_name']

    n_cams = len(camera_positions)

    rgbs = []
    depths = []
    for cam_id in range(n_cams):
        # need to reset everytime you want to take the picture: the camera has mass and it will fall during execution
        env.set_camera(camera_positions[cam_id, :], camera_quats[cam_id, :], camera_name=camera_name)
        rgb, depth = env.render_from_camera(camera_img_height, camera_img_width, camera_name=camera_name)

        rgbs.append(rgb)
        depths.append(depth)

    # return RGBD images of shape [n_cams, h, w, 4]
    return np.concatenate([np.array(rgbs), np.array(depths)[..., None]], -1)


def render_image_from_camX(env, camera_img_height, camera_img_width, camera_fov, camera_name="ext_camera_0"):
    """
    go through all the cameras and get rgbd
    """
    n_cams = 1

    rgbs = []
    depths = []
    pix_T_cams = []
    origin_T_camXs = []
    for cam_id in range(n_cams):
        rgb, depth = env.render_from_camera(camera_img_height, camera_img_width, camera_name=camera_name)
        # need to convert depth to real numbers
        pix_T_camX = get_intrinsics(camera_fov, camera_img_width, camera_img_height)
        origin_T_camX = gymenv_get_extrinsics(env, camera_name)

        rgbs.append(rgb)
        depths.append(depth)
        pix_T_cams.append(pix_T_camX)
        origin_T_camXs.append(origin_T_camX)

    images = dict()
    images['rgb_camXs'] = np.stack(rgbs, axis=0)
    images['depth_camXs'] = np.stack(depths, axis=0)
    images['pix_T_cams'] = np.stack(pix_T_cams, axis=0)
    images['origin_T_camXs'] = np.stack(origin_T_camXs, axis=0)

    return images


def dm_get_extrinsics(physics, cam_id):
    """
        physics: dm_control physics simulator object
        cam_id : id of the camera we want extrinsics for
    """
    pos = physics.data.cam_xpos[cam_id]
    mat = physics.data.cam_xmat[cam_id].reshape(3,3)


    # mujoco z is pointing outwards of the lookat point
    # change it to have z looking at the lookat point

    rot_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    mat = np.dot(mat, rot_mat)

    # rot_mat_adam = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    # rot_mat_adam = np.dot(rot_mat_adam, rot_mat)
    # mat = np.dot(rot_mat_adam, mat)
    ext = np.eye(4)
    ext[:3, :3] = mat
    ext[:3, 3] = pos
    return ext

def gymenv_get_extrinsics(env, cam_name):
    """
        physics: dm_control physics simulator object
        cam_id : id of the camera we want extrinsics for
    """
    #pos, quat = env.get_object_pos(cam_name)
    pos, xmat = env.get_object_pos(cam_name)
    mat = np.zeros((4,4), np.float32)
    mat[3, 3] = 1
    mat[:3, :3] = np.reshape(xmat, [3, 3])
    #pos = physics.data.cam_xpos[cam_id]
    #mat = physics.data.cam_xmat[cam_id].reshape(3,3)
    #mat = transformations.quaternion_matrix(quat)

    # flip y and z
    mat[:,1] *= (-1)
    mat[:,2] *= (-1)
    mat[:3, 3] = pos
    return mat


def get_intrinsics(fovy, img_width, img_height):
    """
        fovy:       fovy supplied in the y-direction
        cam_no:     camera number in the scene
        img_width:  width of the image
        img_height: height of the image
    """
    # fovy = physics.model.cam_fovy[cam_no]
    # now compute the fovx
    # print(f'-- img_width: {img_width} --')
    # print(f'-- img_height: {img_height} --')
    aspect = float(img_width) / img_height
    # assert aspect == 1., "I am giving data such that aspect is 1"
    fovx = 2 * np.arctan(np.tan(np.deg2rad(fovy) * 0.5) * aspect)
    fovx = np.rad2deg(fovx)
    cx = img_width / 2.
    cy = img_height / 2.
    fx = cx / np.tan(np.deg2rad(fovx / 2.))
    fy = cy / np.tan(np.deg2rad(fovy / 2.))
    K = np.zeros((3,3), dtype=np.float)
    K[2][2] = 1
    K[0][0] = fx
    K[1][1] = fy
    K[0][2] = cx
    K[1][2] = cy
    return K


def get_quaternion(z_axis, world_up):
    """
    z_axis = numpy.ndarray(n_pts, 3)
    world_up = axis representing the y axis
    """
    world_up = np.tile(world_up, len(z_axis)).reshape(len(z_axis), 3)
    side_axis = np.cross(world_up, z_axis)
    side_axis = side_axis / np.linalg.norm(side_axis, axis=1).reshape(-1, 1)
    cam_locs_to_remove = np.where(np.isnan(np.linalg.norm(side_axis, axis=1)))
    cam_locs_to_take = np.ones(len(world_up)).astype(np.int)
    cam_locs_to_take[cam_locs_to_remove] = 0
    world_up = world_up[cam_locs_to_take.astype(np.bool)]
    side_axis = side_axis[cam_locs_to_take.astype(np.bool)]
    z_axis = z_axis[cam_locs_to_take.astype(np.bool)]
    up_axis = np.cross(z_axis, side_axis)

    # TODO: find a better way to do this
    rot_mat = np.zeros((len(z_axis), 4, 4))
    quats = list()
    for i in range(len(rot_mat)):
        rot_mat[i, :3, 0] = side_axis[i]
        rot_mat[i, :3, 1] = up_axis[i]
        rot_mat[i, :3, 2] = z_axis[i]
        rot_mat[i, 3, 3] = 1
        if np.isnan(np.sum(rot_mat)):
            print('in the nan of utils while generating quaternions')
            from IPython import embed; embed()
        rot_mat[i] = sym(rot_mat[i])

        quats.append(transformations.quaternion_from_matrix(rot_mat[i]))
    return cam_locs_to_take, np.stack(quats)


def circle_pts(radius, angles):

    xs = radius*np.cos(angles)
    ys = radius*np.sin(angles)
    return np.c_[xs, ys]


def generate_new_cameras_hemisphere(radius, lookat_point, pitch = [20, 60, 20], yaw = [0, 350, 36], yaw_list=None, rot_format="quat"):


    """
    pitch: elevation [min_pitch, max_pitch, d_pitch]
    yaw: azimuth [min_yaw, max_yaw, d_yaw]
    """
    min_pitch, max_pitch, d_pitch = pitch
    min_yaw, max_yaw, d_yaw = yaw

    xyz_points  = []
    quats = []
    if yaw_list == None:
        yaw_list = range(min_yaw, max_yaw + 1, d_yaw)
    for pitch in range(min_pitch, max_pitch + 1, d_pitch):
        for yaw in yaw_list:
            mat_yaw = transformations.euler_matrix(0,0,math.radians(yaw), 'rxyz')
            mat_pitch = transformations.euler_matrix(0,math.radians(-pitch),0, 'rxyz')

            # camera at the origin is x+ = inverse lookat vector(z), y is x of camera
            # camera x: right, camera y: up, camera z: inverse lookat direction
            x_vector = np.zeros((4), dtype=np.float32)
            x_vector[0] = 1

            y_vector = np.zeros((4), dtype=np.float32)
            y_vector[1] = 1

            z_vector = np.zeros((4), dtype=np.float32)
            z_vector[2] = 1

            z_vec_out = mat_yaw.dot(mat_pitch.dot(x_vector))
            x_vec_out = mat_yaw.dot(mat_pitch.dot(y_vector))
            y_vec_out = mat_yaw.dot(mat_pitch.dot(z_vector))


            cam_loc = z_vec_out * radius
            rot_mat = np.eye(4, dtype=np.float32)
            rot_mat[:3, 0] = x_vec_out[:3]
            rot_mat[:3, 1] = y_vec_out[:3]
            rot_mat[:3, 2] = z_vec_out[:3]

            if rot_format == "quat":
                quat = transformations.quaternion_from_matrix(rot_mat)

            else:
                quat = np.reshape(rot_mat[:3,:3], [-1])
            cam_loc = lookat_point + cam_loc[:3]
            xyz_points.append(cam_loc)
            quats.append(quat)

    quats = np.stack(quats, axis=0)
    xyz_points = np.stack(xyz_points, axis=0)

    return xyz_points, quats


def generate_new_cameras(radius, center, lookat_vector, height, jitter_z=False, num_pts=50,
    jitter_amount=0.02):

    """
    radius is the distance to the center on the xy plane
    center[2] is not used
    return:
        xyz_points: nviews x 3
        quat: nviews x 4
    """

    # generate points on the circle
    angle = np.linspace(0, 2*np.pi, num=num_pts) # in radians

    # angle = np.asarray([0, np.pi/2., np.pi, 3*np.pi/2.])
    xy_pts = circle_pts(radius, angle)
    # plt.scatter(xy_pts[:, 0], xy_pts[:, 1], c='b')
    # plt.show()

    # xyz_points
    xyz_points = np.c_[xy_pts[:, 0], xy_pts[:, 1], height*np.ones(len(xy_pts))]
    xyz_points[:, 0] += center[0]
    xyz_points[:, 1] += center[1]
    if jitter_z:
        xyz_points[:, 2] += (jitter_amount*np.random.normal(size=num_pts))

    # generate the z-axis for each of these
    z_vector = xyz_points - lookat_vector
    z_axis = z_vector / np.linalg.norm(z_vector, axis=1).reshape(-1, 1)
    # now from this I will also generate the other two axis and quaternion
    _, quat = get_quaternion(z_axis, world_up=np.asarray([0., 0., 1.]))

    return xyz_points, quat
