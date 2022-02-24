import os
import click
import wandb
import random
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from attrdict import AttrDict
import tensorflow as tf
from tensorflow.data import Dataset
import tensorflow.contrib.slim as slim

from bc.model import SpatialSoftmaxCNN
from dynamics.utils.log_utils import log
from dynamics.utils.data_utils import mkdir
from pcp_utils.utils import config_from_yaml_file


class BehaviorCloningTrainer(object):
    def __init__(self, config):
        self.config = config

        # init logging
        self.log_dir = os.path.join(os.getcwd(), config['log_dir'], 'bc', config['prefix'])
        mkdir(self.log_dir)
        wandb.init(entity='katefgroup',
                   project='quantize',
                   name='bc.{}'.format(config['prefix']),
                   config={k:v for k,v in config.items() if not type(v) is dict},
                   notes=config['notes'],
                   dir=self.log_dir,
                   job_type='train',
                   tags=['bc'])

        self.train()

    def _get_dataset(self, filenames):
        # get number of data points
        num_transitions = 0
        for file in tqdm(filenames, desc='Count dataset samples'):
            with open(file, 'rb') as f:
                num_transitions += len(pickle.load(f))

        # define dataset generator
        def data_generator():
            shuffled_filenames = random.sample(filenames, len(filenames))
            for file_ix, filename in enumerate(shuffled_filenames):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                for transition in data:
                    rgb = transition[0]['images'][..., :3]
                    d = transition[0]['images'][..., [-1]]
                    if self.config['view'] < 0:
                        # get random view
                        view_ix = random.randint(0, len(d) - 1)
                    else:
                        view_ix = self.config['view']
                    rgb = rgb[view_ix]
                    d = d[view_ix]

                    obs = np.concatenate([
                        transition[0]['observation'].flatten(),
                        transition[0]['desired_goal'].flatten()
                    ])
                    gt_aux = transition[0]['observation'].flatten()[3:6]
                    gt_output = transition[1].flatten()

                    yield rgb, d, obs, gt_aux, gt_output

        model_params = self.config['model_params']
        data_shapes = tuple([
            tf.TensorShape([model_params['image_size']] * 2 + [3]),
            tf.TensorShape([model_params['image_size']] * 2 + [1]),
            tf.TensorShape([model_params['obs_dim']]),
            tf.TensorShape([3]),
            tf.TensorShape([model_params['output_dim']])
        ])
        dataset = Dataset.from_generator(data_generator,
                                         output_types=tuple([tf.float32] * 5),
                                         output_shapes=data_shapes)
        dataset = dataset.prefetch(self.config['batch_size'] * 8).repeat()
        dataset = dataset.batch(self.config['batch_size'])
        return dataset, num_transitions

    def _run_single_step(self, session, model, batch, global_step=None,
                         is_train=True, optimizer=None):
        '''Update a single step for the model and return relevant info.'''
        input_data = {
            'input_rgb': batch[0],
            'input_d': batch[1],
            'input_obs': batch[2],
            'gt_aux': batch[3],
            'gt_output': batch[4]
        }
        feed_dict = model.get_feed_dict(input_data)
        fetch_dict = model.get_fetch_dict(is_train=is_train)
        fetch = [global_step, model.output, model.loss] + list(fetch_dict.values())

        if is_train:
            fetch += [optimizer]

        fetched_values = session.run(fetch, feed_dict=feed_dict)
        if is_train:
            fetched_values = fetched_values[:-1]

        step, output, loss = fetched_values[:3]
        info_dict = { k: fetched_values[3+i] for i, k in enumerate(fetch_dict.keys()) }

        return step, output, loss, info_dict, input_data

    def _log_single_step(self, step, output, loss,
                         info_dict, input_data, is_train=True, log_images=False):
        '''Log info for the previous update step in Wandb.'''
        gt = input_data['gt_output']
        pred = output

        prefix = '{}'.format('' if is_train else 'val_')
        if log_images:
            rgb_images = [wandb.Image(img) for img in input_data['input_rgb']]
            depth_images = [wandb.Image(img.astype(float) / np.max(img)) for img in input_data['input_d']]
        log_dict = {
            f'{prefix}step': step,
            f'{prefix}loss': loss,
            f'{prefix}gt_min': np.min(gt),
            f'{prefix}gt_max': np.max(gt),
            f'{prefix}pred_min': np.min(pred),
            f'{prefix}pred_max': np.max(pred)
        }
        if log_images:
            log_dict.update({
                f'{prefix}img_rgb': rgb_images,
                f'{prefix}img_d': depth_images
            })
        log_dict.update({ '{}{}'.format(prefix, k): v for k, v in info_dict.items() })

        wandb.log(log_dict, step=step)

    def train(self):
        config = self.config

        # reset global step and graph
        tf.reset_default_graph()

        # generate dataset from data
        data_files = glob(os.path.join(os.path.expanduser(config['data_dir']),
                                       '*_s.pkl'))
        random.shuffle(data_files)
        num_files = len(data_files)
        num_train = int(config['train_split'] * num_files)
        num_val = int(config['val_split'] * num_files)
        train_files = data_files[:num_train]
        val_files = data_files[num_train:num_train+num_val]
        train_dataset, num_train_samples = self._get_dataset(train_files)
        val_dataset, num_val_samples = self._get_dataset(val_files)
        train_iter = train_dataset.make_initializable_iterator()
        val_iter = val_dataset.make_initializable_iterator()
        train_batch_op = train_iter.get_next()
        val_batch_op = val_iter.get_next()
        log.info(f'Got {num_train_samples} training samples ' + \
                 f'and {num_val_samples} validation samples.')

        # build model
        model_name = config['model_name']
        assert model_name == 'cnn_policy', f'Model {model_name} not supported.'
        model = SpatialSoftmaxCNN(AttrDict(config['model_params']))
        with tf.variable_scope('bc'):
            model.build()

        # check variables
        all_vars = tf.trainable_variables('bc')
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        # optimizer
        global_step = tf.train.create_global_step()
        optimizer = tf.contrib.layers.optimize_loss(
            loss=model.loss,
            global_step=global_step,
            learning_rate=config['learning_rate'],
            optimizer=tf.train.AdamOptimizer
        )

        # saver
        saver = tf.train.Saver(max_to_keep=100)
        pretrain_saver = tf.train.Saver(var_list=all_vars, max_to_keep=1)
        save_dir = os.path.join(self.log_dir, 'ckpt')
        mkdir(save_dir)

        # session
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1}
        )
        session = tf.Session(config=session_config)

        # init variables
        session.run(tf.initialize_all_variables())

        # init dataset iterators
        session.run(train_iter.initializer)
        session.run(val_iter.initializer)

        # training
        num_epochs = config['num_epochs']
        num_steps_per_epoch = num_train_samples // config['batch_size']
        for epoch_ix in tqdm(range(num_epochs), desc='Training Epoch'):
            for step_ix in tqdm(range(num_steps_per_epoch), desc='Step', leave=False):
                # train single step
                batch_train = session.run(train_batch_op)
                step_output = self._run_single_step(session, model,
                                                    batch_train, global_step,
                                                    is_train=True,
                                                    optimizer=optimizer)
                step, output, loss, info_dict, input_data = step_output

                # log training outputs
                if step % config['log_interval'] == 0:
                    self._log_single_step(step, output, loss,
                                          info_dict, input_data,
                                          is_train=True,
                                          log_images=config['log_images'])

                # val single step
                if step % config['val_interval'] == 0:
                    batch_val = session.run(val_batch_op)
                    val_output = self._run_single_step(session, model,
                                                       batch_val, global_step,
                                                       is_train=False)
                    step, val_output, val_loss, val_info_dict, val_input_data = val_output
                    self._log_single_step(step, val_output, val_loss,
                                          val_info_dict, val_input_data,
                                          is_train=False,
                                          log_images=config['log_images'])

            # save checkpoint
            if (epoch_ix + 1) % config['ckpt_interval'] == 0 or epoch_ix == num_epochs - 1:
                ckpt_name = 'ep{:03d}'.format(epoch_ix + 1)
                saver.save(session, os.path.join(save_dir, ckpt_name),
                           global_step=step)


@click.command()
@click.option('--config_file', default='bc/configs/default.yaml',
              help='Path to policy config file.')
@click.option('--prefix', default='default',
              help='Prefix for the run.')
@click.option('--notes', default='',
              help='Notes for the run.')
@click.option('--seed', default=0,
              help='Seed used for dataset split and training.')
def main(config_file, prefix, notes, seed):
    config = config_from_yaml_file(config_file)
    config['prefix'] = prefix
    config['notes'] = notes
    config['model_params']['batch_size'] = config['batch_size']

    np.random.seed(seed)
    random.seed(seed)

    trainer = BehaviorCloningTrainer(config)

    log.info('Finished training.')


if __name__ == '__main__':
    main()
