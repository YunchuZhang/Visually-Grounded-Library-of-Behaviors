# Visually-Grounded-Library-of-Behaviors

## Abstract
  We propose a visually-grounded library of behav-iors  approach  for  manipulating  diverse  objects  across  varyinginitial  and  goal  configurations  and  camera  placements.  Ourkey  innovation  is  to  disentangle  the  standard  image-to-actionmapping  into  two  separate  modules  that  use  different  typesof  perceptual  input:  (1)  a  behavior  selector  which  conditionson  thestaticobject  properties  to  select  the  behaviors  that  cansuccessfully  perform  the  desired  tasks  on  the  object  in  hand,and (2) a library of behaviors each of which conditions on thedynamicobject  properties  to  predict  actions  to  execute  overtime.  The  selector  uses  a  semantically-rich  3D  object  featurerepresentation  extracted  from  images  using  geometry-aware2D-to-3D  neural  networks. 

-------

![Overview](https://github.com/YunchuZhang/Visually-Grounded-Library-of-Behaviors/blob/main/image/overview.png)

-------
# set up

## openAI baseline and gym setup
set up baseline and openai gym path in paths.yaml 
  
  Install the quantize-gym inside the folder
## dataset set up
place trained models under this repo
ex. trained_models/fetch_cup_loss_obs/model_name/..

download object mesh and xml data from Shapenet:
and put it at ../quan_meshes. Remember to add this path to your paths.yaml


## installation:

### baseline env
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

Put gaurav mesh under /projects/katefgroup/gaurav_meshes/meshes
/Users/sfish0101/Documents/2020/Spring/quantize-gym/gym/envs/robotics/assets/stls


if you plan to run with selector using trained features

   pip install munch
   pip install torch==1.4.0
   pip install tensorboardX
   pip install scikit-image
   pip install torchvision
   pip install sklearn

# Run the code
## given a list of initial policies, assign objects to the policies
check command lines in run_compress.sh
    sh run_compress.sh

## generate data for training metric learning
    sh run_data_gen.sh
## metric learning with GRNN
    follow the readme inside pytorch_disco folder
