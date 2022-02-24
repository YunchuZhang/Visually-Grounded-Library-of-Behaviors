import tensorflow as tf
import sys
import numpy as np
import os


import pcp_utils

graspnet_path = pcp_utils.utils.get_6dof_graspnet_dir()
sys.path.append(graspnet_path)


import grasp_estimator



def get_vae_placeholder_dict(args, scope='vae'):
    """
    Returns the dictionary of input tensors that is used for training VAE.

    Args:
      first_dimension: int, it is equal to 
        num_objects_per_batch x num_grasps_per_object.
      args: arguments that are used for training.
      files: list of str, list of training files.
      pcreader: PointCloudReader.
    """
    first_dimension = args.num_objects_per_batch * args.num_grasps_per_object

    OUTPUT_SHAPES = {
        '{}_pc'.format(scope): tf.placeholder(tf.float32, [first_dimension, args.npoints, 4], name = '{}_pc'.format(scope)),
        '{}_grasp_rt'.format(scope): tf.placeholder(tf.float32, [first_dimension, 4, 4], name='{}_grasp_rt'.format(scope)),
        #'{}_pc_pose'.format(scope): tf.placeholder(tf.float32, [first_dimension, 4, 4], name = '{}_pc_pose')
        #'{}_cad_path'.format(scope): tf.placeholder(tf.string, [first_dimension])
        #'{}_cad_scale'.format(scope): tf.placeholder(tf.float32, [first_dimension])
        '{}_quality'.format(scope): tf.placeholder(tf.float32, [first_dimension])
    }

    #    OUTPUT_KEYS = sorted(list(OUTPUT_SHAPES.keys()))
    #    OUTPUT_TYPES = []
    #    for k in OUTPUT_KEYS:
    #        if k == 'cad_path':
    #            OUTPUT_TYPES.append(tf.string)
    #        else:
    #            OUTPUT_TYPES.append(tf.float32)
    return OUTPUT_SHAPES





class GraspTrainer:
    def __init__(self):
        vae_checkpoint_folder = f'{graspnet_path}/checkpoints/latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_'
        evaluator_checkpoint_folder = f'{graspnet_path}/checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_/'

        self.vae_checkpoint_folder = vae_checkpoint_folder
        self.evaluator_checkpoint_folder = evaluator_checkpoint_folder
        self.grasp_conf_threshold = 0.8
        self.gradient_based_refinement = False


        cfg = grasp_estimator.joint_config(
            self.vae_checkpoint_folder,
            self.evaluator_checkpoint_folder,
        )

        cfg['threshold'] = self.grasp_conf_threshold
        cfg['sample_based_improvement'] = 1 - int(self.gradient_based_refinement)
        cfg['num_refine_steps'] = 10 if self.gradient_based_refinement else 20
        cfg['training'] = True
        cfg['logdir'] = "log" 

        cfg["vae_checkpoint_folder"] = None
        cfg["evaluator_checkpoint_folder"] = None
        cfg["npoints"] = 1024
        cfg["num_grasps_per_object"] = 64

        self.cfg = cfg


        self.num_refine_steps = cfg['num_refine_steps']
        self.estimator = grasp_estimator.GraspEstimator(cfg)
        self.sess = tf.Session()

        ph = get_vae_placeholder_dict(cfg)


        self.tf_output = self.estimator.build_network(ph)
        #self.estimator.load_weights(self.sess)
    
    def train(self):
        niter=100

        args = self.cfg
        logdir = os.path.join(args.logdir, 'tf_output')
        if not os.path.isdir(logdir):
            os.makedirs(logdir)



        train_op, summary_op, tf_data_dict, logger_dict, tf_step = self.tf_output
        summary_hook = tf.train.SummarySaverHook(
            summary_op=summary_op,
            output_dir=logdir,
            save_steps=args.save_steps,
        )
    
        logging_hook = tf.train.LoggingTensorHook(
            tensors=logger_dict,
            every_n_iter=args.log_steps,
        )


        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            sess.run(init)

            for _ in range(niter):

                first_dimension = self.cfg["num_objects_per_batch"] * self.cfg["num_grasps_per_object"]
                obj_pc = np.zeros([first_dimension, self.cfg["npoints"], 4], np.float32)
                grasp_rt = np.zeros([first_dimension, 4, 4], np.float32)
                vae_quality= np.ones([first_dimension], np.float32)



                # preparing data
                data_dict = dict()
                data_dict[tf_data_dict["vae_pc"]] = obj_pc
                data_dict[tf_data_dict["vae_grasp_rt"]] = grasp_rt
                data_dict[tf_data_dict["vae_quality"]] = vae_quality

                _, step, logger_out = sess.run([train_op, tf_step, logger_dict], feed_dict=data_dict)
                print("step", step)

                for key in logger_out:
                    if "losses" in key:
                        print(key, logger_out[key])





        #        generated_grasps, generated_scores, _ = self.estimator.predict_grasps(
        #            self.sess,
        #            obj_pc,
        #            latents,
        #            num_refine_steps = self.num_refine_steps,
        #        )




if __name__=="__main__":
    grasp_trainer = GraspTrainer()
    grasp_trainer.train()
