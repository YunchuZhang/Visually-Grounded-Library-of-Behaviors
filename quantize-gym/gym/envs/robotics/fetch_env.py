import numpy as np

from gym.envs.robotics import rotations, robot_env, utils
from gym.envs.robotics.robot_env import DEFAULT_SIZE
from gym.envs.robotics.utils import generate_random_camera_config


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, rim_pts, use_bbox_indicator,
        n_actions, init_info=None, obs_type='xyz', randomize_camera=False,
        obs_image_size=128
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            rim_pts (np.ndarray): contains points which are on the rim of the surface
            use_bbox_indicator (bool): want to display bounding box around the object?
            n_actions (int): 4 or 8 depending on whether you want to use orientation or not default 4
        """
        self.xml_model_path = model_path
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.rim_pts = rim_pts
        self.use_bbox_indicator = use_bbox_indicator
        self.init_info = init_info

        # observation settings
        self.relative_obs = False
        self.include_rotation = True
        self.gripper_pos_in_goal = self.has_object and ('gripper_in_goal' in obs_type)
        self.include_obj_size = 'bb' in obs_type or obs_type == '2d'
        self.include_velocity = False
        self.include_image = '2d' in obs_type
        self.randomize_camera = randomize_camera

        # get possible camera viewpoints if needed
        if self.randomize_camera:
            self._episode_cam_config_id = 0
            self._cam_pos, self.cam_quat = generate_random_camera_config()

        # set default image size
        self._obs_image_size = obs_image_size

        # select one of the rim point to go to
        if self.rim_pts is not None:
            self.chosen_rim_pt = self.rim_pts[-10]
            self.translated_rim_pt = np.zeros(3,)

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions,
            initial_qpos=initial_qpos)

        # now save some joint positions for easy indexing
        if self.use_bbox_indicator:
            self.bbox_joint_qpos_idxs = self.sim.model.get_joint_qpos_addr('bbox_indicator_joint')
            self.bbox_joint_qpos_low, self.bbox_joint_qpos_high = self.bbox_joint_qpos_idxs
            self.bbox_joint_qvel_idxs = self.sim.model.get_joint_qvel_addr('bbox_indicator_joint')
            self.bbox_qvel_low, self.bbox_qvel_high = self.bbox_joint_qvel_idxs

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        if (not action.shape == (8,)) and (not action.shape == (4,)):
            assert False
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if action.shape == (4,):
            mocap_ctrl, gripper_ctrl = action[:3], action[3]
            pos_ctrl, rot_ctrl = mocap_ctrl, [1., 0., 1., 0.]
        elif action.shape == (8,):
            mocap_ctrl, gripper_ctrl = action[:7], action[7]
            pos_ctrl, rot_ctrl = mocap_ctrl[:3], mocap_ctrl[3:7]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = rot_ctrl
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def get_finger_tip_state(self):
        tmp = np.zeros(2)
        for i in range(2): # left, right
            idx = self.sim.model.jnt_qposadr[self.sim.model.actuator_trnid[i, 0]]
            tmp[i] = self.sim.data.ctrl[i]
        return tmp


    def _get_obs(self, obs_arg=None):

        # positions

        #import ipdb; ipdb.set_trace()
        #cam_pos = self.sim.data.get_camera_xpos("ext_camera_0")

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2quat(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
            # everytime the object position might change and based on that the rim point
            # also needs to change. So I will add to the translated rim pt every time
            if self.rim_pts is not None:
                self.translated_rim_pt = self.chosen_rim_pt + object_pos
                gripper_to_rim = self.translated_rim_pt - grip_pos
            else:
                gripper_to_rim = np.zeros(3,)
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric


        # build observation
        obs_parts = []
        if self.relative_obs:
            obs_parts.extend([grip_pos - object_pos.ravel(),
                              object_pos.ravel() - self._init_object_pos])
        else:
            obs_parts.extend([grip_pos, object_pos.ravel(), object_rel_pos.ravel()])

        #
        #        if not self.has_object:
        #            achieved_goal = grip_pos.copy()
        #        else:
        #            achieved_goal = np.squeeze(object_pos.copy())
        obs_parts.append(gripper_state)
        if self.include_rotation: #3
            obs_parts.append(object_rot.ravel())
        if self.include_velocity:
            obs_parts.append(object_velp.ravel())
            if self.include_rotation:
                obs_parts.append(object_velr.ravel())
            obs_parts.extend([grip_velp, gripper_vel])
        if self.include_obj_size: # 3
            obj_size_site_id = self.sim.model.site_name2id('object0:size')
            obj_size = self.sim.model.site_size[obj_size_site_id]
            obs_parts.append(obj_size)
        obs = np.concatenate(obs_parts)


        #        if obs_arg == "25":
        #            obs = np.concatenate([
        #                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        #                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        #            ])
        #        else:
        #            obs = np.concatenate([
        #                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), object_rot.ravel(),
        #            ])
        
        return {
            'observation': obs.copy(),
            'achieved_goal': self.observed_achieved_goal,
            'desired_goal': self.observed_goal,
            'gripper_to_rim': gripper_to_rim.copy(),
        }



    @property
    def achieved_goal(self):
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            if self.gripper_pos_in_goal:
                ag = np.concatenate([
                    grip_pos,
                    np.squeeze(object_pos.copy())
                ])
            else:
                ag = np.squeeze(object_pos.copy())
            return ag
        else:
            grip_pos = self.sim.data.get_site_xpos('robot0:grip')
            return grip_pos.copy()

    @property
    def observed_goal(self):
        if self.relative_obs:
            if self.gripper_pos_in_goal:
                ag = self.goal.copy()
                ag[:3] -= self._init_object_pos
                ag[-3:] -= self._init_object_pos
                return ag
            else:
                return self.goal.copy() - self._init_object_pos
        else:
            return self.goal.copy()


    @property
    def observed_achieved_goal(self):
        if self.relative_obs:
            if self.gripper_pos_in_goal:
                oag = self.achieved_goal
                oag[:3] -= self._init_object_pos
                oag[-3:] -= self._init_object_pos
                return oag
            else:
                return self.achieved_goal - self._init_object_pos
        else:
            return self.achieved_goal

    def set_camera(self, cam_pos, cam_quat, camera_name="ext_camera_0"):
        """
        cam_pos: 3
        cam_quat: 4
        """

        #cam_id = self.sim.model.camera_name2id(camera_name)

        object_qpos = self.sim.data.get_joint_qpos(f'{camera_name}:joint')
        object_qpos[:3] = cam_pos
        object_qpos[3:] = cam_quat
        self.sim.data.set_joint_qpos(f'{camera_name}:joint', object_qpos)
        self.sim.forward()


    def render_from_camera(self, height, width, camera_name):
        self._render_callback()
        rgbd = self.sim.render(width=width, height=height, camera_name=camera_name, depth=True)

        rgb_img = np.flip(rgbd[0], axis=0)
        depth_img = np.flip(rgbd[1], axis=0)
        depth_img = self._convert_depth_to_meters(self.sim, depth_img)

        return rgb_img, depth_img

    def _convert_depth_to_meters(self, sim, depth):
        extent = sim.model.stat.extent
        near = sim.model.vis.map.znear * extent
        far = sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def get_object_pos(self, object_name):
        body_id = self.sim.model.body_name2id(object_name)
        object_xpos = self.sim.data.body_xpos[body_id]
        object_xmat = self.sim.data.body_xmat[body_id]

        return object_xpos, object_xmat


    def get_camera_posrot(self, camera_name):
        cam_qpos = self.sim.data.get_joint_qpos(f'{camera_name}:joint')

        return cam_qpos[:3], cam_qpos[3:]


    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def move_indicator(self, pos, quat=None):
        if self.use_bbox_indicator:
            index = self.bbox_joint_qpos_low
            end_index = self.bbox_joint_qpos_high
            self.sim.data.qpos[index: index+3] = pos
            if quat is not None:
                self.sim.data.qpos[index+3:end_index] = quat
            # finally also make all the velocities zero
            self.sim.data.qvel[self.bbox_qvel_low:] = 0.001

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[site_id]

        #import ipdb; ipdb.set_trace()
        self.sim.forward()

    def sample_pos_from_init_info(self, info):

        object_dpos = np.zeros(3)
        if isinstance(info, str):
            xyz = info.split(" ")
            for idx in range(3):
                if "#" in xyz[idx]:
                    minmax = [float(x) for x in xyz[idx].split("#")]
                    if len(minmax) != 2:
                        print(f"the init format should be float#float but {xyz[idx]} is given")
                    object_dpos[idx] = np.random.uniform(minmax[0], minmax[1], size=1)[0]
                else:
                    object_dpos[idx]  = float(xyz[idx])
        return object_dpos

    def deuler_to_quat(self, euler):
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('zyx', euler, degrees=True)
        quat = rot.as_quat()
        final_quat = np.zeros(4)
        final_quat[1:] = quat[:3]
        final_quat[0] = quat[3]

        return final_quat


    def _reset_sim(self):
        self.sim.set_state(self.initial_state)


        # Randomize start position of object.
        if self.has_object:

            if self.init_info is None:
                object_xpos = self.initial_gripper_xpos[:2]

                # randomize the object around the gripper
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                object_qpos = self.sim.data.get_joint_qpos('object0:joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            else:

                center = np.array(self.init_info["center_of_table"])

                for object_name in self.init_info["objs"]:
                    object_init_info = self.init_info["objs"][object_name]

                    # sample object pos
                    dpos = self.sample_pos_from_init_info(object_init_info['pos'])
                    deuler = self.sample_pos_from_init_info(object_init_info['euler_zyx'])
                    quat = self.deuler_to_quat(deuler)


                    object_qpos = self.sim.data.get_joint_qpos(f'{object_name}:joint')


                    object_qpos[:3] = center + dpos
                    object_qpos[3:] = quat
                    self.sim.data.set_joint_qpos(f'{object_name}:joint', object_qpos)

                    # sample object loc
        self.sim.forward()

        #import ipdb; ipdb.set_trace()

        object_idx = self.sim.model.body_name2id('object0')
        if self.rim_pts is not None:
            self.translated_rim_pt = np.zeros(3,)
            self.translated_rim_pt = self.chosen_rim_pt + self.sim.data.body_xpos[object_idx]
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air:
                goal[2] += self.np_random.uniform(0.1, 0.3)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(1000):
            self.sim.step()


        name = 'object0:joint'
        value = initial_qpos[name]
        #print("====", name,  value)
        self.sim.data.set_joint_qpos(name, value)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
