import os
import os.path as osp
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from gym import error
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]  # changing the position here


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    #    finger_tip_names = ["robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link"]
    #    for i in range(sim.data.ncon):
    #        contact = sim.data.contact[i]
    #        if sim.model.geom_id2name(contact.geom1) in finger_tip_names or \
    #           sim.model.geom_id2name(contact.geom2) in finger_tip_names:
    #            geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
    #            #if np.sqrt(np.sum(np.square(sim.data.cfrc_ext[geom2_body]))) == 0:
    #            #    continue
    #            import ipdb; ipdb.set_trace()
    #
    #            print(sim.model.geom_id2name(contact.geom1),sim.model.geom_id2name(contact.geom2))
    #
    #            print("force:", sim.data.cfrc_ext[geom2_body])
    #            print("force_norm:", np.sqrt(np.sum(np.square(sim.data.cfrc_ext[geom2_body]))))

    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]

def get_points_on_rim(mesh_dir, scale=1.2, obj_name='cup'):
    """This will return me points which lie on the rim of the cup or bowl
    """
    assert os.path.exists(mesh_dir)

    meshes = [osp.join(mesh_dir, m) for m in os.listdir(mesh_dir) if "visual" not in m]
    meshes = [m for m in meshes if "convex" in m]
    if len(meshes) == 0:
        return None
    loaded_meshes = [trimesh.load(m) for m in meshes]
    print(f'applying scale {scale}')
    # scale the mesh
    scaled_meshes = [l.apply_scale(scale) for l in loaded_meshes]

    # combine the meshes
    combined_scaled_mesh = np.sum(scaled_meshes)

    # now get the corners of the bounding box
    bbox_corners = get_bbox_from_mesh(mesh_dir,scale=scale)
    lx, ly, lz = bbox_corners[0]
    rx, ry, rz = bbox_corners[-2]

    # the arrangement of the bounding box is as follows
    # (lx, ly, lz), (rx, ly, lz), (rx, ry, lz), (lx, ry, lz)
    # (lx, ly, rz), (rx, ly, rz), (rx, ry, rz), (lx, ry, rz)
    # the up plane is formed by the following vertices.
    # (2, 3, 6, 7)
    # now I need to sample points in this 2D bounding box
    xs = np.random.uniform(low=lx, high=rx, size=1000)
    ys = np.random.uniform(low=ly, high=ry, size=1000)
    zs = np.random.uniform(low=lz, high=rz, size=1000)

    up_plane = np.c_[xs, ys, np.ones(len(xs))*rz]
    down_plane = np.c_[xs, ys, np.ones(len(xs))*lz]
    left_plane = np.c_[np.ones(len(ys))*lx, ys, zs]
    right_plane = np.c_[np.ones(len(ys))*rx, ys, zs]
    front_plane = np.c_[xs, np.ones(len(xs))*ly, zs]
    back_plane = np.c_[xs, np.ones(len(xs))*ry, zs]

    # plot the mesh and the points, if this is right
    # then I need to find the intersecting points
    up_cloud = trimesh.points.PointCloud(up_plane)
    down_cloud = trimesh.points.PointCloud(down_plane)
    left_cloud = trimesh.points.PointCloud(left_plane)
    right_cloud = trimesh.points.PointCloud(right_plane)
    front_cloud = trimesh.points.PointCloud(front_plane)
    back_cloud = trimesh.points.PointCloud(back_plane)
    scene = trimesh.Scene([combined_scaled_mesh,
        up_cloud, down_cloud, left_cloud, right_cloud,
        front_cloud, back_cloud])
    # scene.show()

    # now compute the distance of all the points on the up-plane
    # to the mesh surface
    closest_points, distances, triangle_id = combined_scaled_mesh.nearest.on_surface(
            up_plane)

    pts_idx = distances < 3e-4
    filtered_points = closest_points[pts_idx]
    # draw the spheres, of small radius
    spheres_list = list()
    for p in filtered_points:
        tmesh = trimesh.creation.icosphere(radius=0.003,
                color=np.asarray([1, 0, 0]).astype(np.uint8))
        # apply the translation
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = p
        tmesh = tmesh.apply_transform(trans_mat)
        spheres_list.append(tmesh)
    
    # draw it on my friend
    scene = trimesh.Scene([combined_scaled_mesh]+[spheres_list])
    # scene.show()
    return filtered_points

def generate_random_camera_config(radius=0.5, lookat_point=[1.3, 0.75, 0.4],
                                  pitch=[40, 41, 2], yaw=[0, 350, 36],
                                  yaw_list=[300, 0, 60], rot_format="quat"):
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

