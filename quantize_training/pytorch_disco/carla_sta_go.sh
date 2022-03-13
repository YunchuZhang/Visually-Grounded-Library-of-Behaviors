#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_STA GO"
echo "-----"

MODE="CARLA_STA"
export MODE
# CUDA_VISIBLE_DEVICES=1 
python -W ignore main.py

echo "----------"
echo "CARLA_STA GO DONE"
echo "----------"

