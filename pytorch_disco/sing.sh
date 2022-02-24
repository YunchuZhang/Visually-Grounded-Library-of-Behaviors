#!/bin/bash

echo ""
echo "loading the new driverless image!"
IMG="/projects/data/singularity/ubuntu-16.04-lts-tensorflow-gpu-1.8.0-python-pytorch-0.4.0.img"
ls -alht $IMG
module load singularity
singularity shell -B /projects:/projects --nv $IMG
echo ""
echo "exited the singularity image!"
echo ""
