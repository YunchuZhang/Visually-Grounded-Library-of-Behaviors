import os
import numpy as np

np.random.seed(0)

dataset_dir = "/home/gauravp/new_closeup_dataset"
objects = os.listdir(dataset_dir)
train_file_handler = open(os.path.join("/home/gauravp/train_files",
    'close_up_touch_and_visual.txt'), 'w')

for obj in objects:
    instances = os.listdir(os.path.join(dataset_dir, obj))
    # now for each instance get its visual file and write to a directory
    for inst in instances:
        fp = os.path.join(os.path.join(dataset_dir, obj), inst)
        visual_path = os.path.join(fp, 'visual_data.npy')
        touch_path = os.path.join(fp, 'touch_data.npy')
        if not os.path.exists(visual_path):
            raise FileNotFoundError('check the visual path again')
        if not os.path.exists(touch_path):
            raise FileNotFoundError('check the touch path again')

        # I modify the touch dataset here, it will now contain two more
        # fields named "train_touch_idxs" and "val_touch_idxs".
        # 10 % of all the data is used for validation.
        touch_data = np.load(touch_path, allow_pickle=True).item()
        size_for_this_obj = len(touch_data['sensor_imgs'])
        print('number of touch points for this object are: {}'.format(size_for_this_obj))
        train_len = int(0.9 * size_for_this_obj)
        perm = np.random.permutation(size_for_this_obj)
        train_idxs = perm[:train_len]
        val_idxs = perm[train_len:]
        # just a simple test to quell my paranoia
        for v in val_idxs:
            assert v not in train_idxs, "val in train not allowed brother"

        touch_data['train_idxs'] = train_idxs
        touch_data['val_idxs'] = val_idxs

        np.save(touch_path, touch_data)
        
        train_file_handler.write(f'{visual_path},{touch_path}\n')
        # train_file_handler.write(f'{touch_path}\n')
    
train_file_handler.close()