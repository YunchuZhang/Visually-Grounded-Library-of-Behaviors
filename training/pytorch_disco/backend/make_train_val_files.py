import numpy as np
import argparse
import os
from random import shuffle


def make_train_val(dirs, save_path, train_percent=80):
    all_files = list()
    for d in dirs:
        for f in os.listdir(d):
            all_files.append(os.path.join(d, f))

    shuffle(all_files) 
    
    num_train_files = int((80 / 100.) * len(all_files))
    num_val_files = len(all_files) - num_train_files

    # split the train and val files
    train_files = all_files[:num_train_files]
    val_files = all_files[num_train_files:]

    assert len(train_files) == num_train_files
    assert len(val_files) == num_val_files

    # write the files finally
    train_file = open(os.path.join(save_path, "train_file_controller_dt1542020.txt"), "w")
    val_file = open(os.path.join(save_path, "val_file_controller_dt1542020.txt"), "w")

    for f in train_files:
        train_file.write(f'{f}\n')
    train_file.close()

    for f in val_files:
        val_file.write(f'{f}\n')
    val_file.close()

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str,
            help='npy files containing base dir')
    parser.add_argument('--save_path', type=str,
            help='save train and val files here')
    
    args = parser.parse_args()
    dirs = [os.path.join(args.dir_path, d) for d in os.listdir(args.dir_path)]
    print(dirs)

    make_train_val(dirs, args.save_path, train_percent=80)
