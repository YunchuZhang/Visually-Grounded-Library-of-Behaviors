# Run the experiments for training 3d tensor using affordance labels

python main.py [MODEL NAME] -- exp_name=[exp name in your exp config file] --run_name=[a name for this run]

training with view prediction

python -W ignore main.py MUJOCO_OFFLINE --exp_name=train_viewpred_occ  --run_name=viewpred_occ_using_grasptop_all




# load model
We provide our pre-trained model and raw real data [here](https://github.com/user/repo/blob/branch/other_file.md).  
You may need to put the data and raw folder in quantize_training folder, and put checkpoints folder in pytorch_disco folder.

if specified load path with loadname, then it will use the specified loadname
otherwise it will use the latest model in the current folder

to use loadname:
model: the total model, "./checkpoints/exp_name/run_name" or "./checkpoints/exp_name/run_name/model-step.pth"

# Preprocess real data
Please check vis.py and transfer.py


