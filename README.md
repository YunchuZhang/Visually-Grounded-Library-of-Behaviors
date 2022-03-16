# Visually-Grounded-Library-of-Behaviors
[Project Website](https://yunchuzhang.github.io/vbes.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://openreview.net/forum?id=sIVC-oZN1PQ)&nbsp;&nbsp;•&nbsp;&nbsp;Conference on Robot Learning (CoRL) 2021

## Abstract
We propose a method for manipulating diverse objects across a wide range of initial and goal configurations and camera placements. We disentangle the standard image-to-action mapping into two separate modules: (1) a behavior selector which conditions on intrinsic and semantically-rich object appearance features to select the behaviors that can successfully perform the desired tasks on the object in hand, and (2) a library of behaviors each of which conditions on extrinsic and abstract object properties to predict actions to execute over time. Our framework outperforms various learning and non-learning based baselines in both simulated and real robot tasks. 



![Overview](https://github.com/YunchuZhang/Visually-Grounded-Library-of-Behaviors/blob/main/images/overview.png)


# Installation
**Step 1.** Recommended: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3.7.

```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u
echo $'\nexport PATH=~/miniconda3/bin:"${PATH}"\n' >> ~/.profile  # Add Conda to PATH.
source ~/.profile
conda init
```

**Step 2.** Create and activate Conda environment, then install GCC and Python packages.

```shell
conda create --name vbe python=3.7 -y
conda activate vbe
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
conda env update --file vbe.yaml
```

**Step 3.** Set up related packages path and download our shared [data&mesh](https://drive.google.com/drive/folders/1eXLVr1F2R_fVTkgsXuQzM_0lQP36gp_w?usp=sharing) 
```shell

## gym environment setup
Install the vbe-gym inside the folder by running 
pip install -e.
change gym path inside paths.yaml to the vbe-gym path

After that, installing the following packages:

pip install trimesh (for gym)
pip install dm_control
pip install open3d==0.8.0.0

# install mujoco
# put the license at ~/.mujoco/mjkey.txt
pip install install -U 'mujoco-py<2.1,>=2.0'

conda install libgcc
conda install -c conda-forge mesalib
conda install -c menpo osmesa
conda install -c anaconda mesa-libgl-cos6-x86_64ex


## dataset set up 

download object mesh and xml data from Shapenet:
and put it at ../quan_meshes. Remember to add this path to your paths.yaml
Put our preprocessed mesh stl files to /gym/envs/robotics/assets/stls
```

# Run the code
**Step 1.1** Generating training data and affordance label for the training set in simulation.
```
sh run_data_gen.sh
sh run_compress.sh
```

**Step 1.2** Processing real data to the same pkl format and visualize them.
```shell
python training/transfer.py
python training/vis.py
```



**Step 2** Training our model.
```
python main.py [MODEL NAME] -- exp_name=[exp name in your exp config file] --run_name=[a name for this run]

eg.training with 3D affordance model

python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=plate_0612_load --run_name=plate_0612_r2_cluttered

eg.training with 2D affordance model

python -W ignore main.py MUJOCO_OFFLINE_METRIC_2D --exp_name=plate_0612_load --run_name=train_2d_0827

for more usage, please check other commands in training/run.sh

```
**Step 3** Evaluating model .
```
python -W ignore main.py MUJOCO_OFFLINE_METRIC --exp_name=test_train_plate_0612_load --run_name=test_train_plate_0612_load
```


# Citation 
```bibtex
@inproceedings{yang2021visually,
  title={Visually-Grounded Library of Behaviors for Manipulating Diverse Objects across Diverse Configurations and Views},
  author={Yang, Jingyun and Tung, Hsiao-Yu and Zhang, Yunchu and Pathak, Gaurav and Pokle, Ashwini and Atkeson, Christopher G and Fragkiadaki, Katerina},
  booktitle={5th Annual Conference on Robot Learning},
  year={2021}
}
```
