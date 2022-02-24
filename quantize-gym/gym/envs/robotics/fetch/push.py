import os
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', xml_path=None, rim_pts=None,
                 use_bbox_indicator=False, n_actions=4, init_info=None, obs_type='xyz'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        if xml_path is not None:
            global MODEL_XML_PATH
            MODEL_XML_PATH = xml_path

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, rim_pts=None,
            use_bbox_indicator=use_bbox_indicator, n_actions=n_actions, init_info=init_info,
            obs_type=obs_type)
        utils.EzPickle.__init__(self)

