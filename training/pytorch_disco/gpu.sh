#!/bin/bash

# # v100 gpus:
# srun -p KATE_RESERVED --time=72:00:00 --gres gpu:1 -c10 --mem=63g --nodelist=compute-0-14 --pty $SHELL

# any gpu
srun -p KATE_RESERVED --time=72:00:00 --gres gpu:1 -c4 --mem=28g --exclude=compute-0-[22,36,38,20] --pty $SHELL

# srun --time=24:00:00 --gres gpu:1 -c4 --mem=28g --pty $SHELL
