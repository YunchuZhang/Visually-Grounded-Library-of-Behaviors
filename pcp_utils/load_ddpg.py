import json
import sys, os

import gym

import numpy as np
import tensorflow as tf
import multiprocessing



#from pcp_utils.utils import rescale_mesh_in_env

CACHED_ENVS = {}

def cached_make_env(make_env):
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]

def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()
    env_name = kwargs['env_name']

    def make_env(subrank=None):
        env = gym.make(env_name)
        max_episode_steps = env._max_episode_steps
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')


    kwargs['T'] = tmp_env._max_episode_steps

    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def load_policy(load_path, model_name, obs_arg=None):
    import baselines.her.experiment.config as config
    from baselines.common import tf_util
    import tensorflow as tf

    with open(os.path.join(load_path, 'logs/params.json'), 'r') as f:
        params = json.load(f)

    clip_return=True

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        params = prepare_params(params)
        dims = config.configure_dims(params)
        if obs_arg == "25":
            dims['o'] = 25
        policy = config.configure_ddpg(dims=dims, params=params, reuse = False,clip_return=clip_return)
        if load_path is not None:
            model_path = os.path.join(load_path, 'models')
            model_filename = os.path.join(model_path , model_name)
            tf_util.load_variables(model_filename)
            print("Successfully loaded a policy.")
    return policy

def main():
    load_path="/Users/apokle/Documents/quantized_policies/trained_models_fetch/fetch_cup/159e56c18906830278d8f8c02c47cde0/models/save55"
    params_path="/Users/apokle/Documents/quantized_policies/trained_models_fetch/fetch_cup/159e56c18906830278d8f8c02c47cde0/logs"

    model = load_policy(load_path, params_path)

    model_xml_path = 'pick_and_place_cup_159e56c18906830278d8f8c02c47cde0.xml'

    mesh = '159e56c18906830278d8f8c02c47cde0'
    # Initialize a fetch Pick and Place environment
    #rescale_mesh_in_env(mesh, scale=1.2)
    env = gym.make("FetchPickAndPlace-v1", model_xml_path=model_xml_path)

    obs = env.reset()


    episode_rew = 0
    i = 0
    while True:

        actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        # print(actions)
        episode_rew += rew
        env.render()
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            obs = env.reset()
        i+=1

    env.close()

    return model

if __name__ == '__main__':
    # while True:
    main()
        # tf.get_variable_scope().reuse_variables()