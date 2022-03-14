# Visually-Grounded-Library-of-Behaviors
[Project Website](https://yunchuzhang.github.io/vbes.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://openreview.net/forum?id=sIVC-oZN1PQ)&nbsp;&nbsp;•&nbsp;&nbsp;Conference on Robot Learning (CoRL) 2021

## Abstract
We propose a method for manipulating diverse objects across a wide range of initial and goal configurations and camera placements. We disentangle the standard image-to-action mapping into two separate modules: (1) a behavior selector which conditions on intrinsic and semantically-rich object appearance features to select the behaviors that can successfully perform the desired tasks on the object in hand, and (2) a library of behaviors each of which conditions on extrinsic and abstract object properties to predict actions to execute over time. Our framework outperforms various learning and non-learning based baselines in both simulated and real robot tasks. 



![Overview](https://github.com/YunchuZhang/Visually-Grounded-Library-of-Behaviors/blob/main/images/overview.png)


# Installation

## openAI baseline and gym setup
set up baseline and openai gym path in paths.yaml 
  
  Install the quantize-gym inside the folder
## dataset set up
place trained models under this repo
ex. trained_models/fetch_cup_loss_obs/model_name/..

download object mesh and xml data from Shapenet:
and put it at ../quan_meshes. Remember to add this path to your paths.yaml




### baseline env setup
```shell
conda create --name py36-baselines python=3.6
pip install tensorflow-gpu==1.14

In baseline: python setup.py install
In gym: pip install -e .

pip install pyyaml
pip install ipdb
pip install trimesh (for gym)
pip install matplotlib

pip install dm_control
pip install open3d==0.8.0.0
pip install transformations
pip install ordered_set
pip install attrdict
pip install bounding-box
pip install addict
pip install click
pip install click

# install mujoco
# put the license at ~/.mujoco/mjkey.txt
pip install install -U 'mujoco-py<2.1,>=2.0'

conda install libgcc


conda install -c conda-forge mesalib
conda install -c menpo osmesa
conda install -c anaconda mesa-libgl-cos6-x86_64ex
```
Put our preprocessed mesh under [mesh](https://drive.google.com/drive/folders/1eXLVr1F2R_fVTkgsXuQzM_0lQP36gp_w?usp=sharing) 
to quantize-gym/gym/envs/robotics/assets/stls


if you plan to run with selector using trained features
```shell
 pip install munch
 pip install torch==1.4.0
 pip install tensorboardX
 pip install scikit-image
 pip install torchvision
 pip install sklearn
```
# Run the code
## given a list of initial policies, assign objects to the policies
check command lines in run_compress.sh
    sh run_compress.sh

## generate data for training metric learning
    sh run_data_gen.sh
## metric learning with GRNN
    please follow the readme inside pytorch_disco folder and commands in quantize_training/run.sh
# Train with real dataset
## installation:
    please follow the commands inside quantize_training/run.sh
# Citation 
```bibtex
@inproceedings{yang2021visually,
  title={Visually-Grounded Library of Behaviors for Manipulating Diverse Objects across Diverse Configurations and Views},
  author={Yang, Jingyun and Tung, Hsiao-Yu and Zhang, Yunchu and Pathak, Gaurav and Pokle, Ashwini and Atkeson, Christopher G and Fragkiadaki, Katerina},
  booktitle={5th Annual Conference on Robot Learning},
  year={2021}
}
```
