import numpy as np
import tensorflow as tf
from attrdict import AttrDict

from policy.policy import Policy
from pcp_utils.utils import Config
from bc.model import SpatialSoftmaxCNN as SSCNN
from pcp_utils.utils import config_from_yaml_file as parse_yaml
from pcp_utils.load_ddpg import get_session
from pcp_utils.rollouts import simple_rollouts

class CNNPolicy(Policy):
    class Config(Config):
        policy_name = 'cnn_policy'
        policy_model_path = None
        params_path = None
        camera_config_path = None
        viewpoint = None

    def __init__(self, config: Config):
        self.config = config
        self.policy_name = self.config.policy_name

        # get model
        model_params = parse_yaml(self.config.params_path)['model_params']
        model_params['batch_size'] = 1
        self.model = SSCNN(AttrDict(model_params))
        with tf.variable_scope('bc'):
            self.model.build()
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope='bc')

        # get session
        self.session = get_session()

        # restore model
        loader = tf.train.Saver(var_list=model_vars)
        loader.restore(self.session, self.config.policy_model_path)

    def step(self, obs):
        batch_data = {
            'input_rgb': obs['images'][0, :, :, :3][None],
            'input_d': obs['images'][0, :, :, 3:][None],
            'input_obs': np.concatenate([
                obs['observation'].flatten()[None],
                obs['desired_goal'].flatten()[None]
            ], axis=-1),
            'gt_aux': np.array([[0, 0, 0]]),
            'gt_output': np.array([[0, 0, 0, 0]])
        }
        feed_dict = self.model.get_feed_dict(batch_data)
        ac = self.session.run(self.model.output, feed_dict=feed_dict).flatten()
        return ac, None, None, None

    def run_forwards(self, env, num_rollouts, path_length, return_obs=False,
                     prepare_data=False, image_size=None, num_visualize=1,
                     detector=None):
        assert prepare_data == False, 'Prepare data is not impelmented yet.'
        assert detector is None, 'Detector is not supported yet.'
        camera_config = parse_yaml(self.config.camera_config_path)
        camera_config['viewpoint'] = self.config.viewpoint
        camera_config = AttrDict(camera_config)
        stats = simple_rollouts(env, self,
                                num_rollouts=num_rollouts,
                                path_length=path_length,
                                return_obs=return_obs,
                                prepare_data=prepare_data,
                                image_size=image_size,
                                num_visualize=num_visualize,
                                detector=detector,
                                camera_config=camera_config)
        return stats
