import os
import os.path as osp

DATA_PATH = "/projects/katefgroup/gauravp/touch_project/depth_cam_env/data_with_depth_patches/data"
all_objects = os.listdir(DATA_PATH)

visual_paths = [osp.join(DATA_PATH, obj_dir, f'{obj_dir}_128_128.pkl')
    for obj_dir in all_objects if not obj_dir.startswith('.')]
sensor_paths = [osp.join(DATA_PATH, obj_dir, f'{obj_dir}_sensor_readings.pkl')
    for obj_dir in all_objects if not obj_dir.startswith('.')]

assert all([osp.exists(visual_path) for visual_path in visual_paths])
assert all([osp.exists(sensor_path) for sensor_path in sensor_paths])

# now write to a file
train_file_path = "/projects/katefgroup/gauravp/touch_project/depth_cam_env/data_with_depth_patches/curr_full_train.txt"
file_handler = open(train_file_path, 'w')
# write based on the objects
for obj_name in all_objects:
    for v, s in zip(visual_paths, sensor_paths):
        if obj_name in v and obj_name in s:
            file_handler.write(f'{v}\n{s}\n')

file_handler.close()
