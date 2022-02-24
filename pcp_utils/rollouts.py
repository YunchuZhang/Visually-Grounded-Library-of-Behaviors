import numpy as np
import cv2
import sys
import getpass
# sys.path.append("/home/{}".format(getpass.getuser()))
# from discovery.backend.mujoco_online_inputs import get_inputs
import tensorflow as tf


# from mpc.utils.video_utils import add_caption_to_img
from pcp_utils.cameras import generate_new_cameras_hemisphere as gen_cameras
from pcp_utils.cameras import render_images_from_config


EXPERT_KEYS = [ 'observation',
                'desired_goal',
                'achieved_goal',
                ]

baseline_short_keys = [ 'observation',
                'desired_goal',
                'achieved_goal']

short_keys = [  'observation',
                'observation_with_orientation',
                'observation_abs',
                'desired_goal',
                'desired_goal_abs',
                'achieved_goal',
                'achieved_goal_abs',
                'image_observation',
                'object_pose',
                # 'object_pos',
                # 'image_desired_goal',
                # 'image_achieved_goal',
                'depth_observation',
                # 'depth_desired_goal',
                'cam_info_observation',
                ]
                # 'cam_info_goal']



def evaluate_rollouts(paths):
    """Compute evaluation metrics for the given rollouts."""
    total_returns = [path['rewards'].sum() for path in paths]
    episode_lengths = [len(p['rewards']) for p in paths]

    diagnostics = OrderedDict((
        ('return-average', np.mean(total_returns)),
        ('return-min', np.min(total_returns)),
        ('return-max', np.max(total_returns)),
        ('return-std', np.std(total_returns)),
        ('episode-length-avg', np.mean(episode_lengths)),
        ('episode-length-min', np.min(episode_lengths)),
        ('episode-length-max', np.max(episode_lengths)),
        ('episode-length-std', np.std(episode_lengths)),
    ))

    return diagnostics

def convert_to_active_observation(x):
    flattened_observation = np.concatenate([
        x[key] for key in EXPERT_KEYS], axis=-1)
    return [flattened_observation[None]]

def return_stats(rewards, count_infos, goal_reach_percentage):
    return {'min_return': np.min(rewards),
            'max_return': np.max(rewards),
            'mean_return': np.mean(rewards),
            'mean_final_success': np.mean(count_infos),
            'success_rate': goal_reach_percentage}

# obs = np.concatenate([
#     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
#     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
# ])


def simple_rollouts(env, policy, scale=1, num_rollouts=100, path_length=50,
                    rescale_obs=False, return_obs=True,
                    prepare_data=False, image_size=None, num_visualize=1,
                    collect_data_fn=None, detector=None, camera_config=None):
    avg_reward = 0
    success_rate = 0
    is_success = []
    paths = []
    imgs = []
    actions = []

    # setup camera config
    if camera_config is not None:
        cam_pos, cam_quat = gen_cameras(radius=camera_config.radius,
                                        lookat_point=camera_config.lookat_pos,
                                        pitch=camera_config.pitch,
                                        yaw=camera_config.yaw,
                                        yaw_list=camera_config.yaw_list)
        if 'viewpoint' in camera_config:
            view_ix = camera_config['viewpoint']
            cam_pos = cam_pos[view_ix][None]
            cam_quat = cam_quat[view_ix][None]
        camera_config['pos'] = cam_pos
        camera_config['quat'] = cam_quat

    for sample_ix in range(num_rollouts):
        obs = env.reset()
        if return_obs:
            path = [obs]

        if camera_config is not None:
            images = render_images_from_config(env.unwrapped, camera_config)
            obs['images'] = images

        if sample_ix < num_visualize or prepare_data:
            actions.append([])
            imgs.append([])
            if image_size is not None:
                img = env.render(mode='rgb_array', height=image_size,
                                 width=image_size)
            else:
                img = env.render(mode='rgb_array')
            if not prepare_data:
                img = add_caption_to_img(img, { 't': 0 },
                                         name=policy.policy_name)
            imgs[-1].append(img)

        cur_reward = 0
        for t in range(path_length):
            # substitute object position with detector output
            if detector is not None:
                detector_output = detector.predict_forward(env)
                print(obs)
                import pdb; pdb.set_trace()

            ac, _, _, _ = policy.step(obs)
            obs, rew, done, info = env.step(ac)
            if return_obs:
                path.append(obs)

            if rescale_obs:
                obs = rescale_observations(obs, scale=scale)

            if camera_config is not None:
                images = render_images_from_config(env.unwrapped, camera_config)
                obs['images'] = images

            if sample_ix < num_visualize or prepare_data:
                actions[-1].append(ac)
                if image_size is not None:
                    img = env.render(mode='rgb_array', height=image_size,
                                     width=image_size)
                else:
                    img = env.render(mode='rgb_array')
                if not prepare_data:
                    img_info = { 't': t + 1 }
                    img_info.update(info)
                    img = add_caption_to_img(img, img_info,
                                             name=policy.policy_name)
                imgs[-1].append(img)

            if collect_data_fn is not None:
                collect_data_fn(env, t)

            cur_reward += rew

        if cur_reward > -1 * path_length:
            # only append successful paths
            if return_obs and len(path) >= 2:
                paths.append(path)
            success_rate += 1
            is_success.append(1)
        else:
            is_success.append(0)

        avg_reward += cur_reward
        print("Reward {}".format(cur_reward))

    success_rate /= num_rollouts

    return {
        'avg_reward': avg_reward / num_rollouts,
        'success_rate': success_rate,
        'is_success': np.array(is_success),
        'paths': paths,
        'imgs': np.array(imgs),
        'actions': np.array(actions)
    }


def rollout(env,
            num_rollouts,
            path_length,
            policy,
            expert_policy=None,
            mesh = None,
            is_test=False,
            is_init_data=False,
            scale=1.0):

    env_keys = baseline_short_keys
    puck_z = 0.01

    #import pdb; pdb.set_trace()
    # Check instance for DDPG
    # <baselines.her.ddpg.DDPG object at 0x7f70a8560e10>
    if str(policy).find('DDPG')!=-1:
        # actions, _, _, _ = model.step(obs)
        # actor = policy.actions_np
        actor = policy.step
    else:
        actor = policy.act
        observation_converter = lambda x: x

    if expert_policy:
        assert str(expert_policy).find('DDPG')!=-1
        expert_actor = expert_policy.step

    paths = []
    rewards = []
    count_infos = []
    img = 0
    count = 0
    while len(paths) < (num_rollouts):
        print(len(paths))
        t = 0
        path = {key: [] for key in env_keys}
        images = []
        infos = []
        observations = []
        actions = []
        terminals = []
        obj_sizes = []
        puck_zs = []
        observation = env.reset()

        # print("Before rescaling obs ", observation["observation"])
        # observation["observation"][5] *= scale
        # print("After rescaling obs ", observation["observation"])
        # cv2.imwrite('store/{}.png'.format(img),goal_img)
        # img = img + 1

        first_reward = True
        R = 0
        for t in range(path_length):
            # observation = observation_converter(observation)
            if str(policy).find('DDPG')!=-1:
                action,_,_,_ = actor(observation)
            else:
                action = actor(observation)

            if expert_policy:
                # exp_observation = exp_observation_converter(observation)
                expert_action,_,_,_ = expert_actor(observation)
            else:
                expert_action = action

            observation, reward, terminal, info = env.step(action)

            # print("Before rescaling obs ", observation["observation"])
            # observation["observation"][5] *= scale
            # print("After rescaling obs ", observation["observation"])

            # image = env.render(mode='rgb_array') #cv2 show image
            # cv2.imwrite('store/'+'{}_'.format(mesh)+'{}.png'.format(img),image)
            # img = img + 1

            for key in env_keys:
                path[key].append(observation[key])
            actions.append(expert_action)
            terminals.append(terminal)

            infos.append(info)
            R += reward

            if reward == 0 and first_reward:
                count += 1
                print("Episode Reward=", R, reward)
                first_reward = False

            # if terminal:
            #   print('episode_rew={}'.format(R))

            #   if isinstance(policy, GaussianPolicy):
            #       policy.reset()
            #   break
        #------------------------------------------------------------------------------
        # If R == path_length, the episode very likely failed; do not append such paths
        # Stopping this step for now --------------------------------------------------
        print("Reward ", R)
        #if is_test or (is_init_data and (R > -path_length)) or (len(actions) > 0):
        path = {key: np.stack(path[key], axis=0) for key in env_keys}
        path['actions'] = np.stack(actions, axis=0)
        path['terminals'] = np.stack(terminals, axis=0).reshape(-1,1)

        if isinstance(policy, GaussianPolicy) and len(path['terminals']) >= path_length:
            continue
        elif not isinstance(policy, GaussianPolicy) and len(path['terminals'])==1:
            continue

        rewards.append(R)
        count_infos.append(infos[-1]['puck_success'])
        paths.append(path)

    print('Minimum return: {}'.format(np.min(rewards)))
    print('Maximum return: {}'.format(np.max(rewards)))
    print('Mean return: {}'.format(np.mean(rewards)))
    print('Mean final success: {}'.format(np.mean(count_infos)))
    print('Goal Reach Percentage: {}'.format(count/num_rollouts))
    return _clean_paths(paths), return_stats(rewards, count_infos, count/num_rollouts)

def _clean_paths(paths):
    """Cleaning up paths to only contain relevant information like
       observation, next_observation, action, reward, terminal.
    """

    clean_paths = {key: np.concatenate([path[key] for path in paths]) for key in paths[0].keys()}

    return clean_paths

def append_paths(main_paths, paths):
    if main_paths is None or len(main_paths) == 0:
        return paths
    if len(paths) == 0:
        return main_paths
    """ Appending the rollouts obtained with already exisiting data."""
    paths = {key: np.vstack((main_paths[key], paths[key])) for key in main_paths.keys()}
    return paths
