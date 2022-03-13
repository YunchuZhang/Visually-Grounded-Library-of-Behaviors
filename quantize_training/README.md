# Run the experiments for training 3d tensor using affordance labels

python main.py [MODEL NAME] -- exp_name=[exp name in your exp config file] --run_name=[a name for this run]

training with view prediction

python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_viewpred_occ  --run_name=viewpred_occ_using_grasptop_all




# load model
if specified load path with loadname, then it will use the specified loadname
otherwise it will use the latest model in the current folder

to use loadname:
model: the total model, "./checkpoints/exp_name/run_name" or "./checkpoints/exp_name/run_name/model-step.pth"


old run
sh ./bacjend/run_train.sh


# First-time setup

Install cuda corr3d

```
cd cuda_ops/corr3d
python setup.py install
cd ..
```

Install tensorboardX

`pip install tensorboardX`

Install moveipy:

`pip install moviepy`

Install Scikit-image:

`pip install scikit-image`

Run

`./carla_sta_go.sh`

# Tensorboard

With some cpu resources, on say node `0-16`, run tensorboard with a command like `./tb.sh 3456`.

On the head node, open a tunnel to your tensorboard node, with a command like `./tunnel.sh 3456 0-16`.

# Development

To develop new features and ideas, you will usually run things in `CUSTOM` mode. Run `./custom_go.sh` to do this. If you do not have an `exp_custom.py` file, you should create one. You can copy from any other experiments file. For example, you may want to start with `cp exp_carla_sta.py exp_custom.py`.

Note that `exp_custom.py` is in the `.gitignore` for this repo! This is because custom experiments are private experiments -- and do not usually last long. Once you get a solid result that you would like others to be able to reproduce, add it to one of the main `exp_whatever.py` files and push it to the repo.
