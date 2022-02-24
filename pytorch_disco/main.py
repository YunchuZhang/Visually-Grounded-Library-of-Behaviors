
import click
import os
import cProfile
import logging
import torch
#from model_carla_sta import CARLA_STA
# from model_carla_flo import CARLA_FLO
import model_nets
import utils


logger = logging.Logger('catch_all')
@click.command()
@click.argument("mode", required=True)
@click.option("--exp_name", default="trainer_basic", help="execute expriment name defined in config")
@click.option("--run_name", default="1", help="run name")


def main(mode, exp_name, run_name):

    os.environ["MODE"] = mode
    os.environ["exp_name"] = exp_name
    os.environ["run_name"] = run_name

    import config_files.hyperparams as config

    config.run_name = run_name
    config.exp_name = exp_name
    config.mode = mode
    config.run_full_name = f'{mode}/{run_name}'

    config.log_dir = f"log/{mode}/{run_name}"
    config.checkpoint_dir = f"checkpoints/{mode}/{run_name}"
    config.vis_dir = f"vis/{mode}/{run_name}"
    
    utils.utils.makedir(config.log_dir)
    # only create checkpoint directory if it is training
    if config.train_mode == "train":
        utils.utils.makedir(config.checkpoint_dir)


    try:
        if config.do_carla_sta:
            model = model_nets.CARLA_STA(config)
            model.go()

        elif config.do_carla_flo:
            model = model_nets.CARLA_FLO(config)
            model.go()

        elif config.do_mujoco_offline:
            model = model_nets.MUJOCO_OFFLINE(config)
            model.go()

        elif config.do_mujoco_offline_metric:
            model = model_nets.MUJOCO_OFFLINE_METRIC(config)
            model.go()

        elif config.do_mujoco_offline_metric_2d:
            model = model_nets.MUJOCO_OFFLINE_METRIC_2D(config)
            model.go()

        elif config.do_touch_embed:
            # initializer for touch embed is called
            model = model_nets.TOUCH_EMBED(config)
            # till here the dataloader is created for all the sets_to_run
            model.go()
        else:
            assert(False) # what mode is this?

    except (Exception, KeyboardInterrupt) as ex:
        logger.error(ex, exc_info=True)
        log_cleanup(log_dir_)

def log_cleanup(log_dir_):
    log_dirs = []
    for set_name in hyp.set_names:
        log_dirs.append(log_dir_ + '/' + set_name)

    for log_dir in log_dirs:
        for r, d, f in os.walk(log_dir):
            for file_dir in f:
                file_dir = os.path.join(log_dir, file_dir)
                file_size = os.stat(file_dir).st_size
                if file_size == 0:
                    os.remove(file_dir)

if __name__ == '__main__':
    main()
    # cProfile.run('main()')

