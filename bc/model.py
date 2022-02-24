import tensorflow as tf
import tensorflow.contrib.slim as slim


def spatial_soft_argmax(x):
    ''' Implementation of the spatial softmax layer
        in Deep Spatial Autoencoders for Visuomotor Learning.
        See paper at https://arxiv.org/pdf/1509.06113.pdf.
        Code inspired by https://github.com/tensorflow/tensorflow/issues/6271.

    Args:
        x: input of shape [N, H, W, C]
    '''
    N, C, H, W = x.get_shape().as_list()

    # convert softmax to shape [N, H, W, C, 1]
    features = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [N * C, H * W])
    softmax = tf.nn.softmax(features)
    softmax = tf.transpose(tf.reshape(softmax, [N, C, H, W]), [0, 2, 3, 1])
    softmax = tf.expand_dims(softmax, -1)

    # get image coordinates in shape [H, W, 1, 2]
    posx, posy = tf.meshgrid(tf.lin_space(-1., 1., num=H),
                             tf.lin_space(-1., 1., num=W),
                             indexing='ij')
    image_coords = tf.stack([posx, posy], -1)
    image_coords = tf.expand_dims(image_coords, 2)

    # return argmax coordinates of [N, C * 2]
    res = tf.reduce_sum(softmax * image_coords, reduction_indices=[1, 2])
    res = tf.reshape(res, [N, C * 2])
    return res


class SpatialSoftmaxCNN(object):
    def __init__(self, model_config):
        self.batch_size = model_config.batch_size
        self.image_size = model_config.image_size
        self.obs_dim = model_config.obs_dim
        self.rgb_layers = model_config.rgb_layers
        self.depth_layers = model_config.depth_layers
        self.cnn_layers = model_config.cnn_layers
        self.aux_task = model_config.aux_task
        self.fc_layers = model_config.fc_layers
        self.output_min = model_config.output_min
        self.output_max = model_config.output_max
        self.output_dim = model_config.output_dim
        self.l1_loss_weight = model_config.l1_loss_weight
        self.l2_loss_weight = model_config.l2_loss_weight
        self.aux_loss_weight = model_config.aux_loss_weight

        self.output_translate = (self.output_max + self.output_min) / 2
        self.output_scale = (self.output_max - self.output_min) / 2

        if self.aux_task == 'obj_pos':
            self.aux_task_dim = 3
        else:
            raise ValueError(f'Undefined aux task {self.aux_task}.')

        # create placeholders
        self.input_rgb = tf.placeholder(name='input_rgb', dtype=tf.float32,
                                        shape=[self.batch_size,
                                               self.image_size,
                                               self.image_size,
                                               3])
        self.input_d = tf.placeholder(name='input_d', dtype=tf.float32,
                                      shape=[self.batch_size, self.image_size,
                                             self.image_size, 1])
        self.input_obs = tf.placeholder(name='input_obs', dtype=tf.float32,
                                        shape=[self.batch_size, self.obs_dim])
        self.gt_aux = tf.placeholder(name='gt_aux', dtype=tf.float32,
                                     shape=[self.batch_size, self.aux_task_dim])
        self.gt_output = tf.placeholder(name='gt_output', dtype=tf.float32,
                                        shape=[self.batch_size, self.output_dim])

    def get_feed_dict(self, batch):
        return {
            self.input_rgb: batch['input_rgb'],
            self.input_d: batch['input_d'],
            self.input_obs: batch['input_obs'],
            self.gt_aux: batch['gt_aux'],
            self.gt_output: batch['gt_output']
        }

    def get_fetch_dict(self, is_train=True):
        return {
            'loss_l2': self.l2_loss,
            'loss_l1': self.l1_loss,
            'loss_aux': self.aux_loss
        }

    def build(self):
        rgb_embed = slim.stack(self.input_rgb, slim.conv2d,
                               self.rgb_layers, scope='rgb_embed')
        d_embed = slim.stack(self.input_d, slim.conv2d,
                             self.depth_layers, scope='d_embed')
        _ = tf.concat([rgb_embed, d_embed], axis=-1, name='concat_rgbd')
        _ = slim.stack(_, slim.conv2d, self.cnn_layers, scope='cnn')

        _ = spatial_soft_argmax(_)

        self.aux_output = slim.fully_connected(_, self.aux_task_dim, scope='aux')

        _ = tf.concat([_, self.aux_output, self.input_obs], -1)
        _ = slim.stack(_, slim.fully_connected, self.fc_layers, scope='fc')
        _ = slim.fully_connected(_, self.output_dim, activation_fn=tf.tanh,
                                 scope='fc_out')
        _ = (_ + self.output_translate) * self.output_scale
        self.output = _
 
        self.l2_loss = tf.reduce_mean((self.output - self.gt_output) ** 2)
        self.l1_loss = tf.reduce_mean(tf.abs(self.output - self.gt_output))
        self.aux_loss = tf.reduce_mean((self.aux_output - self.gt_aux) ** 2)
        self.loss = self.l2_loss * self.l2_loss_weight + \
                    self.l1_loss * self.l1_loss_weight + \
                    self.aux_loss * self.aux_loss_weight
