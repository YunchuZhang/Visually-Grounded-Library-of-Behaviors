import json
import copy
# import tensorflow as tf
import numpy as np
import torch
import munch
import time
from os import getpid
from torch.utils.data import DataLoader
# from backend import readers
import os
import skimage.transform
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
#from utils_basic import *
from collections import defaultdict
import random
from multiprocessing import Pool


import utils

#class CarlaTFRecordDataLoader(DataLoader):
#    def __init__(self, dataset):
#        super().__init__()
#        self.dataset = dataset

def parse_intrinsics(int_mat):
    return int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2]

def vectorized_unproject(depth, intrinsics, rgb=None, depth_scale=1., depth_trunc=1000.):
    assert depth.ndim == 2, "Please remove the batch dim or channel dim if you have those"
    assert intrinsics.ndim == 2, "Please remove the batch dim if you have one"
    fx, fy, cx, cy = parse_intrinsics(intrinsics)
    # first scale the entire depth image
    depth /= depth_scale

    # form the mesh grid
    xv, yv = np.meshgrid(np.arange(depth.shape[1], dtype=float), np.arange(depth.shape[0], dtype=float))

    xv -= cx
    xv /= fx
    xv *= depth
    yv -= cy
    yv /= fy
    yv *= depth
    points = np.c_[xv.flatten(), yv.flatten(), depth.flatten()]

    if rgb is not None:
        # flatten it and add to the points
        rgb = rgb.reshape(-1, 3)
        points = np.concatenate((points, rgb), axis=1)
    return points


# class TFRecordDataset():
# 
#     #def __init__(self, dataset_path):
#     #    dataset = tf.data.TFRecordDataset(dataset_path)
#     #    self.dataset = dataset
#     #    iter = self.dataset.make_one_shot_iterator()
# 
#     def __init__(self, dataset_path, shuffle=True, val=False):
#         with open(dataset_path) as f:
#             content = f.readlines()
#         records = [hyp.dataset_location + '/' + line.strip() for line in content]
#         nRecords = len(records)
#         self.nRecords = nRecords
#         print('found %d records in %s' % (nRecords, dataset_path))
#         nCheck = np.min([nRecords, 1000])
#         for record in records[:nCheck]:
#             assert os.path.isfile(record), 'Record at %s was not found' % record
#         print('checked the first %d, and they seem to be real files' % (nCheck))
# 
#         dataset = tf.data.TFRecordDataset(
#             records,
#             compression_type="GZIP"
#         ).repeat()
# 
#         if val:
#             num_threads = 1
#         else:
#             num_threads = 4
# 
#         if hyp.dataset_name=='carla' or hyp.dataset_name=='kitti':
#             dataset = dataset.map(readers.carla_parser,
#                                   num_parallel_calls=num_threads)
#         else:
#             assert(False) # reader not ready yet
# 
#         if shuffle:
#             dataset = dataset.shuffle(buffer_size=100)
#         dataset = dataset.batch(hyp.B)
#         self.dataset = dataset
#         self.iterator = None
#         self.sess = tf.Session()
# 
#         self.iterator = self.dataset.make_one_shot_iterator()
#         self.batch_to_run = self.iterator.get_next()
# 
#     def __getitem__(self, index):
#         try:
#             batch = self.sess.run(self.batch_to_run)
#         except tf.errors.OutOfRangeError:
#             self.iterator = self.dataset.make_one_shot_iterator()
#             self.batch_to_run = self.iterator.get_next()
#             batch = self.sess.run(self.batch_to_run)
# 
#         batch_torch = []
#         for b in batch:
#             batch_torch.append(torch.tensor(b))
# 
#         d = {}
#         [d['pix_T_cams'],
#          d['cam_T_velos'],
#          d['origin_T_camRs'],
#          d['origin_T_camXs'],
#          d['rgb_camRs'],
#          d['rgb_camXs'],
#          d['xyz_veloXs'],
#          d['boxes3D'],
#          d['tids'],
#          d['scores'],
#          ] = batch_torch
# 
#         if hyp.do_time_flip:
#             d = random_time_flip_batch(d)
# 
#         return d
# 
#     def __len__(self):
#         return 10000000000 #never end

class TouchEmbedData(torch.utils.data.Dataset):
    def __init__(self, dataset_path, plot=False, set_name=None):
        """
        dataset path comes from hyp.sets_to_run: hyp.dataset_path
        it consists of visual file in the first line and touch file in the second for different
        objects
        """
        with open(dataset_path, 'r') as f:
            lines = f.readlines()

        # some checks
        self.num_files = len(lines)  # because first line will be visual data and second line will be sensor data
        self.all_files = lines
        self.set_name = set_name
        if self.set_name == 'val':
            self.num_val_samples = 200
            hyp.sensor_S = self.num_val_samples
            hyp.emb_3D_num_samples = self.num_val_samples
        # assert (len(lines) % 2) == 0, "This should be equal to zero,\
        #         else you have not specified both files for one object"

        # every record here is a dict which will return data related to one object
        # self.records = self._preprocess_data(lines)

        # I will pre-emptively load all the numpy files in the dictionary
        self.all_data_dict = defaultdict(list)
        self.data_dir_paths = list()
        self.object_names = list()
        for record in self.all_files:
            record = record.strip('\n')
            visual_file, sensor_file = record.split(',')
            assert os.path.exists(visual_file)
            assert os.path.exists(sensor_file)

            visual_data = np.load(visual_file, allow_pickle=True).item()
            sensor_data = np.load(sensor_file, allow_pickle=True).item()

            d, o = self.get_data_dir(visual_file)
            self.data_dir_paths.append(d)
            self.object_names.append(o)
            print('Done loading for object is {}'.format(o))
            self.all_data_dict[o] = [visual_data, sensor_data]

    @staticmethod
    def generate_pts_from_depths(length, depths, pix_T_cams, origin_T_camXs, origin_T_camRs):
        V = hyp.V  ## (128 x 128)
        xyz_camXs = torch.zeros((length, np.prod(depths[0].shape), 3))
        xyz_camRs = torch.zeros((length, np.prod(depths[0].shape), 3))

        for s in range(length):
            depth = torch.unsqueeze(depths[s], axis=0)
            single_intrinsic = torch.unsqueeze(pix_T_cams[s], axis=0)
            xyz_cam = utils.geom.depth2pointcloud(depth, single_intrinsic,
                    device=torch.device('cpu')).squeeze()

            assert len(xyz_cam) == np.prod(depth.shape), "this should be V as no clipping is done"

            xyz_camXs[s] = xyz_cam
            camR_T_camX = torch.mm(torch.inverse(origin_T_camRs[s]), origin_T_camXs[s])
            xyz_camR = utils.geom.apply_4x4(camR_T_camX.unsqueeze(0), xyz_cam.unsqueeze(0)).squeeze(0)
            xyz_camRs[s] = xyz_camR

        return xyz_camXs, xyz_camRs

    @staticmethod
    def get_data_dir(filename):
        """
            just returns the directory path where touch/visual.npy file
            is located at
        """
        splits = filename.split('/')
        just_object_dir = splits[-2]
        dir_path = splits[:-1]
        dir_path = '/'.join([d for d in dir_path])
        return dir_path, just_object_dir

    def _preprocess_data(self, record,
        data_dir_path, object_name, save_to_dir=False):
        """
            record : python_list
            Assuming the record contains two elements one being
            visual and other being touch. this is already loading
        """
        visual_data = record[0]
        sensor_data = record[1]

        num_visual_cameras = len(visual_data.rgb_camXs)
        print('Length visual cameras = {}'.format(num_visual_cameras))
        num_touch_locations = len(sensor_data.sensor_imgs)
        print('Number of touch locations = {}'.format(num_touch_locations))

        if hyp.do_touch_forward:
            # filepaths
            cam_locs_path = os.path.join(data_dir_path, 'cam_positions.npy')
            cam_quats_path = os.path.join(data_dir_path, 'cam_quat.npy')
            assert os.path.exists(cam_locs_path), "hmmm no cam locations"
            assert os.path.exists(cam_quats_path), "hmm no cam quats here"
            cam_locs = np.load(cam_locs_path)
            cam_quats = np.load(cam_quats_path)

            # number of cameras
            assert len(cam_locs) == len(cam_quats), "missing some orientations not good"
            num_touch_cam_locations = len(cam_locs)

            # len touch locations better be equal to num touch locations
            assert num_touch_locations == num_touch_cam_locations, "did you not include some of the sensor images"

        # now take every tenth-image for making the 3d-visual tensor
        idxs = [j for j in range(num_visual_cameras) if j % 10 == 0]
        idxs = idxs[1:]  # taking every tenth image to form the visual tensor

        if save_to_dir:
            fig, axes = plt.subplots(1, len(idxs), sharex=True, sharey=True)
            for w, idx in enumerate(idxs):
                axes[w].imshow(visual_data.rgb_camXs[idx])

            fig.savefig(f'/home/gauravp/pytorch_disco/input_save/chosen_imgs_{object_name}.png')

        # convert to torch compatible format
        chosen_vis_rgbCamXs = np.transpose(visual_data.rgb_camXs[idxs], [0, 3, 1, 2])
        assert chosen_vis_rgbCamXs.shape[1] == 3, "channel dimension should be along axis=1 for torch"
        chosen_vis_rgbCamRs = np.transpose(visual_data.rgb_camRs[idxs], [0, 3, 1, 2])
        assert chosen_vis_rgbCamRs.shape[1] == 3, "channel dimension should be along axis=1 for torch"
        chosen_vis_depthCamXs = visual_data.depth_camXs[idxs]
        chosen_vis_intrinsics = visual_data.intrinsics[idxs]
        chosen_vis_extrinsics = visual_data.extrinsics[idxs]
        origin_T_camR = np.linalg.inv(visual_data.camR_T_origin)
        origin_T_camRs = np.reshape(origin_T_camR, [1, 4, 4])
        origin_T_camRs = np.tile(origin_T_camRs, [len(idxs), 1, 1])

        # saves the images camR, camX, and depths from visual cameras
        if save_to_dir:
            # this is to check if all the images are correct, rgb_camXs, depth_camXs and rgb_camRs
            fig, axes = plt.subplots(len(chosen_vis_rgbCamXs), 3, sharex=True, sharey=True)
            for w in range(len(chosen_vis_rgbCamXs)):
                # I believe need to change to numpy and plot it
                temp_x = np.transpose(chosen_vis_rgbCamXs[w], [1, 2, 0])
                temp_r = np.transpose(chosen_vis_rgbCamRs[w], [1, 2, 0])
                temp_d = chosen_vis_depthCamXs[w]
                axes[w][0].imshow(temp_x)
                axes[w][1].imshow(temp_d)
                axes[w][2].imshow(temp_r)

            fig.savefig(f'/home/gauravp/pytorch_disco/input_save/chosen_all_{object_name}.png')

        # convert everything to torch and get the pointclouds too
        chosen_vis_images = torch.from_numpy(chosen_vis_rgbCamXs).float()
        chosen_vis_depths = torch.from_numpy(chosen_vis_depthCamXs).float().unsqueeze(1)
        pix_T_cams = torch.from_numpy(chosen_vis_intrinsics).float()
        origin_T_camXs = torch.from_numpy(chosen_vis_extrinsics).float()
        origin_T_camRs = torch.from_numpy(origin_T_camRs).float()
        chosen_vis_rgbCamR = torch.from_numpy(chosen_vis_rgbCamRs).float()

        a = [torch.allclose(origin_T_camRs[0], origin_T_camRs[k]) for k in range(len(origin_T_camRs))]
        assert all(a), "this should all be true for ref cam"
        b = [torch.allclose(pix_T_cams[0], pix_T_cams[k]) for k in range(len(pix_T_cams))]
        assert all(b), "this should all be true for pix_T_cam"

        xyz_camXs, xyz_camRs = self.generate_pts_from_depths(len(idxs), chosen_vis_depths,
            pix_T_cams, origin_T_camXs, origin_T_camRs)

        # now put it in a dict and you are good to go for visual_features
        chosen_vis_images = chosen_vis_images / 255.
        chosen_vis_images -= 0.5
        chosen_vis_rgbCamR = chosen_vis_rgbCamR / 255.
        chosen_vis_rgbCamR -= 0.5

        d = dict()
        d['rgb_camXs'] = chosen_vis_images
        d['rgb_camRs'] = chosen_vis_rgbCamR
        d['depth_camXs'] = chosen_vis_depths
        d['pix_T_cams'] = pix_T_cams
        d['origin_T_camXs'] = origin_T_camXs
        d['origin_T_camRs'] = origin_T_camRs
        d['xyz_camXs'] = xyz_camXs
        d['xyz_camRs'] = xyz_camRs

        if save_to_dir:
            save_path = f'/home/gauravp/pytorch_disco/input_save/camr_xyz_{object_name}.npy'
            new_xyz_camRs = xyz_camRs.reshape(-1, 3)
            np.save(save_path, new_xyz_camRs)

        # many similar things are required to be done for the sensor file too
        # now if this is train set you need to sample from the train_idxs else
        # from the val_idxs but also keep in mind that for val the number of images
        # are not very high, so you need different number, I do 100 here
        if self.set_name == 'train':
            # Instead of this random permutation I will fix it 1024 select the first 1024
            # perm = np.random.permutation(sensor_data.train_idxs)
            # chosen_idxs = perm[:hyp.sensor_S]
            chosen_idxs = sensor_data.train_idxs[:hyp.sensor_S]
        elif self.set_name == 'val':
            perm = np.random.permutation(sensor_data.val_idxs)
            chosen_idxs = perm[:self.num_val_samples]
        else:
            print('I do not know which data to load')
            assert False
        sensor_imgs = np.transpose(sensor_data.sensor_imgs[chosen_idxs], [0, 3, 1, 2])
        sensor_depths = sensor_data.sensor_depths[chosen_idxs]
        sensor_intrinsics = sensor_data.sensor_intrinsics[chosen_idxs]
        sensor_extrinsics = sensor_data.sensor_extrinsics[chosen_idxs]
        if hyp.do_touch_forward:
            chosen_cam_locs = cam_locs[chosen_idxs]
            chosen_cam_quats = cam_quats[chosen_idxs]
        sensor_origin_T_camR = np.linalg.inv(sensor_data.camR_T_origin)
        sensor_origin_T_camRs = np.reshape(sensor_origin_T_camR, [1, 4, 4])
        sensor_origin_T_camRs = np.tile(sensor_origin_T_camRs, [len(sensor_imgs), 1, 1])

        ## ... Depth resizing and clipping ... ##
        # now I would like rescale the depth images and the sensor intrinsics as well
        my_resize = lambda img: skimage.transform.resize(img, (16, 16), anti_aliasing=True)
        resized_sensor_depths = np.stack(list(map(my_resize, sensor_depths)))
        # now that you have resized the depth images, you should also resize the intrinsics
        scale_x, scale_y = 16 / sensor_depths[0].shape[1], 16 / sensor_depths[0].shape[0]
        sensor_intrinsics = utils.geom.scale_intrinsics(sensor_intrinsics, scale_x, scale_y, py=True)

        # NOTE: this next thing you need to discuss with Adam, I am clipping the depth values, assigning zeros to depth
        # greater than 0.075
        invalid_idxs = np.where(resized_sensor_depths >= 0.075)
        resized_sensor_depths[invalid_idxs[0], invalid_idxs[1], invalid_idxs[2]] = 0.0

        # some data-specific assertions
        assert resized_sensor_depths.max() < 0.075, "I set all thing above 0.075 to zero"
        assert resized_sensor_depths.min() >= 0.0, "the z for depth images is positive"

        if save_to_dir:
            # I will save some of the images and depths fromt the sensor too for sanity!!
            fig_size = 2 * np.asarray([10, 2])
            fig, axes = plt.subplots(3, 16, figsize=fig_size)
            for u, w in enumerate(range(0, 16)):
                axes[0][u].imshow(np.transpose(sensor_imgs[w], [1,2,0]))
                axes[1][u].imshow(sensor_depths[w])
                axes[2][u].imshow(resized_sensor_depths[w])

            fig.savefig(f'/home/gauravp/pytorch_disco/input_save/sensor_imgs_{object_name}.png')
        ## ... Depth resizing and clipping end ... ##

        # convert to torch and then again reconstruct everything in camR
        sensor_imgs = torch.from_numpy(sensor_imgs).float()
        sensor_depths = torch.from_numpy(resized_sensor_depths).float().unsqueeze(1)
        sensor_intrinsics = torch.from_numpy(sensor_intrinsics).float()
        sensor_extrinsics = torch.from_numpy(sensor_extrinsics).float()
        sensor_origin_T_camRs = torch.from_numpy(sensor_origin_T_camRs).float()
        if hyp.do_touch_forward:
            chosen_cam_locs = torch.from_numpy(chosen_cam_locs).float()
            chosen_cam_quats = torch.from_numpy(chosen_cam_quats).float()

        # do some checking
        a = [torch.allclose(sensor_origin_T_camRs[0], sensor_origin_T_camRs[k]) for k in range(len(sensor_origin_T_camRs))]
        assert all(a), "this should all be true for ref cam"
        b = [torch.allclose(sensor_intrinsics[0], sensor_intrinsics[k]) for k in range(len(sensor_intrinsics))]
        assert all(b), "this should all be true for pix_T_cam"

        sensor_xyz_camXs, sensor_xyz_camRs = self.generate_pts_from_depths(len(sensor_imgs), sensor_depths,
            sensor_intrinsics, sensor_extrinsics, sensor_origin_T_camRs)

        if save_to_dir:
            save_path = f'/home/gauravp/pytorch_disco/input_save/sensor_camr_xyz_{object_name}.npy'
            new_sensor_camRs = sensor_xyz_camRs.reshape(-1, 3)
            np.save(save_path, new_sensor_camRs)

        # just to satisfy my paranoia I again check the dimensionality of the sensor depths
        assert sensor_depths.max() < 0.075, "this i did above"
        assert sensor_depths.min() >= 0.0, "this i also did above and assigned the value"

        # now convert the images from (-0.5, 0.5) and you are done
        sensor_imgs = sensor_imgs / 255. - 0.5
        d['sensor_imgs'] = sensor_imgs
        d['sensor_depths'] = sensor_depths
        d['sensor_intrinsics'] = sensor_intrinsics
        d['sensor_extrinsics'] = sensor_extrinsics
        d['sensor_origin_T_camRs'] = sensor_origin_T_camRs
        d['sensor_xyz_camXs'] = sensor_xyz_camXs
        d['sensor_xyz_camRs'] = sensor_xyz_camRs
        if hyp.do_touch_forward:
            d['sensor_locs'] = chosen_cam_locs
            d['sensor_quats'] = chosen_cam_quats
        d['object_name'] = object_name

        return d

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # choose one of the dict based on the record
        chosen_record = list(self.all_data_dict.keys())[idx]  # remember you are choosing from one of the records
        record_vals = self.all_data_dict[chosen_record]
        ddir_path, obj_name = self.data_dir_paths[idx], self.object_names[idx]
        assert isinstance(record_vals, list)
        assert len(record_vals) == 2
        assert ddir_path.split('/')[-1] == obj_name == chosen_record
        data_dict = self._preprocess_data(record_vals, data_dir_path=ddir_path,
            object_name=obj_name)

        # check that all keys that I want are present
        keys_required = ['rgb_camXs', 'rgb_camRs', 'depth_camXs', 'pix_T_cams',
            'origin_T_camXs', 'origin_T_camRs', 'xyz_camXs', 'sensor_imgs',
            'sensor_depths', 'sensor_intrinsics', 'sensor_extrinsics', 'sensor_origin_T_camRs',
            'sensor_xyz_camXs', 'sensor_xyz_camRs']

        for k in keys_required:
            assert k in list(data_dict.keys()), f"key {k} is missing in the record"

        return data_dict


class MuJoCoOfflineData(torch.utils.data.Dataset):
    def __init__(self, config, dataset_path, plot=False, train=True, fix_view=False, num_workers=1, ndata=None, preprocess_on_batch=False):
        # hyperparameters
        self.S = config.S
        self.V = config.V
        self.fixed_view = fix_view
        self.fixed_view_list = [0, 1, 2, 3] #0, 0] #[10, 14, 19, 2, 7]
        self.preprocess_on_batch = preprocess_on_batch
        # open list of files for training
        folder_root_dir = utils.utils.get_source_dir()
        print("folder_root_dir", folder_root_dir)
        print("dataset", dataset_path)

        self.dataset_path = dataset_path
        with open(os.path.join(dataset_path), 'r') as f:
            all_files = f.readlines()


        data_root_dir = utils.utils.get_data_dir()
        records = [os.path.join(data_root_dir, line.strip('\n')) for line in all_files]

        if ndata is not None:
            random.shuffle(records)
            records = records[:ndata]

        nrecords = len(records)
        print(f'----- found {nrecords} in {dataset_path} ----------')
        for record in records:
            assert os.path.isfile(record), f"Record {record} was not found"

        self.device = torch.device('cuda')
        self.prep_train = train


        start_time = time.time()

        if not self.preprocess_on_batch:
        
            if num_workers > 1:
    
                records_split = np.array_split(records, num_workers)
                arg_list = []
                for i in range(num_workers):
                    arg_list.append([records_split[i], True])
                print("start pool")
                with Pool(num_workers) as p:
                    results = p.starmap(self._preprocess, arg_list)
                #pool = Pool(20)
                #results = pool.starmap(self._preprocess, arg_list)
                #pool.close() 
                #pool.join()
                print("end pool")
                self.records = []
                for item in results:
                    self.records += item
            else:
                self.records = self._preprocess(records)
        else:
            self.records = records

        #        print("fast load done", time.time() - start_time)
        #        start_time = time.time()
        #        self.records = self._preprocess(records)
        #
        #        print("slow load done", time.time() - start_time)
        #        import ipdb; ipdb.set_trace()


        print(f'----- actual generated data size {len(self.records)} --------')

        ## what did the _preprocess do
        # I have lets say 5 records and each record has 51 values for each key
        # Say my hyp.S = 10, I am breaking each record into 5 parts
        # so a total of 25 parts, so basically the size of the records should be 25

        # now if the data is validation, take out only one file belonging to each cups
        #if not self.prep_train:
        #    # only take one set of views of a scene
        #    idxs = range(0, len(self.records), 50 // config.S)
        #    filtered_records = list()
        #    for i in idxs:
        #        filtered_records.append(self.records[i])

        # #    self.records_tmplen = filtered_records
        #    print(f'Length of the records: {len(self.records)}')



        self.plot = plot
        if self.plot:
            # create a temp directory to save images and depths
            dump_dir = utils.utils.get_dump_dir()
            self.temp_dir = os.path.join(dump_dir, "temp_im_data")

            utils.utils.makedirs(self.temp_dir)
            self.color_dir = f'{self.temp_dir}/colors'
            self.depth_dir = f'{self.temp_dir}/depths'
            utils.utils.makedirs(self.color_dir)
            utils.utils.makedirs(self.depth_dir)


    def _preprocess(self, records, is_print=False):
        if is_print:
            print("start processing", getpid())
        seq_records = list()
        V = self.V
        
        for record_id, record in enumerate(records):

            if is_print:
                if record_id % 500 == 0:
                    print("running records", getpid(), record_id, "/", len(records))

            # I have checked this above but again doing it here
            if not os.path.exists(record):
                raise FileNotFoundError(f'{record} not found')

            data = np.load(record, allow_pickle=True).item()

            """
            rgb_camXs: nviews x 128 x 128 x 3
            depth_camXs: nviews x 128 x 128
            pix_T_cams: nviews x 3 x3
            camR_T_camXs: nviews x 4 x 4
            bbox_camR: 8x3
            cluster_id: string
            """

            rgb_camXs = data.rgb_camXs[:-1]
            depth_camXs = data.depth_camXs[:-1]
            # last image is shoot from ref cam

            camR_T_camXs = data.camR_T_camXs[:-1]
            pix_T_cams = data.pix_T_cams[:-1]
            origin_T_camR = np.eye(4, dtype=np.float32) #np.linalg.inv(data.camR_T_origin)
            origin_T_camRefs = data.camR_T_camXs[-1:]
            rgb_camRefs = data.rgb_camXs[-1:]


            # now ref_frame_bbox should also be added to the mix
            bbox_in_ref_cam = data.bbox_camR
           

            if "cluster_id" in data:
                #if "cluster_id" in data
                cluster_id = data.cluster_id
            else:
                cluster_id = 0

            # below I am breaking 51 images into S equal parts randomly
            # sample 10 times

            #print("here1", getpid(), record_id, "/", len(records))

            rgbs = np.transpose(rgb_camXs, [0,3,1,2])
            depths = depth_camXs

            rgb_refs = np.transpose(rgb_camRefs, [0,3,1,2])

            origin_T_camXs = camR_T_camXs
            pix_T_cam = pix_T_cams
            num_views = len(rgbs)
            # this is actually identities
            origin_T_camRs = np.reshape(origin_T_camR, [1, 4, 4])
            origin_T_camRs = np.tile(origin_T_camRs, [num_views , 1, 1])

            rgbs = torch.from_numpy(rgbs).float()
            rgb_refs = torch.from_numpy(rgb_refs).float()
            depths = torch.from_numpy(depths).float().unsqueeze(1)
            pix_T_cam = torch.from_numpy(pix_T_cam).float()
            origin_T_camXs = torch.from_numpy(origin_T_camXs).float()
            origin_T_camRs = torch.from_numpy(origin_T_camRs).float()

            if "success_rates_over_class" in data.keys():
                success_rates = torch.from_numpy(np.array(data.success_rates_over_class)).float()
            else:
                success_rates = torch.from_numpy(np.zeros((30), np.float32))

            #print("here2", getpid(), record_id, "/", len(records))

            xyz_camXs = utils.geom.depth2pointcloud(depths, pix_T_cam, device=torch.device('cpu'))

            #print("here3", getpid(), record_id, "/", len(records))

            camR_T_camX = origin_T_camXs
            xyz_camRs = utils.geom.apply_4x4(camR_T_camX, xyz_camXs)

            if len(bbox_in_ref_cam.shape) == 2:
                bbox_in_ref_cam = np.expand_dims(bbox_in_ref_cam, axis=0)
                cluster_id = [cluster_id]
            bbox_in_ref_cam = torch.from_numpy(bbox_in_ref_cam).float()

            #print("here4", getpid(), record_id, "/", len(records))

            d = dict()
            rgbs = rgbs / 255.
            rgb_refs = rgb_refs / 255.
            rgbs = rgbs - 0.5
            rgb_refs = rgb_refs - 0.5

            d['rgb_camXs'] = rgbs # numseq x 3 x 128 x 12
            d['rgb_camRs'] = rgb_refs # 1 x 3 x 128 x 128
            d['depth_camXs'] = depths # numseq x 1 x 128 x 128
            d['pix_T_cams'] = pix_T_cam # numseq x 3 x 3
            d['origin_T_camXs'] = origin_T_camXs # numseq x 4 x 4
            d['origin_T_camRs'] = origin_T_camRs # numseq x 4 x 4
            d['origin_T_camRefs'] = origin_T_camRefs

            d['xyz_camXs'] = xyz_camXs # numseq x V x 3
            d['xyz_camRs'] = xyz_camRs # numseq x V x 3
            d['bbox_in_ref_cam'] = bbox_in_ref_cam # Nobjs x 8 x3
            d['cluster_id'] = cluster_id # string
            d['record'] = record # string, record_path
            d['success_rates'] = success_rates


            #print("here5", getpid(), record_id, "/", len(records))
            seq_records.append(d)
        if is_print:
            print("end processing", getpid())
        return seq_records



    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # select the i-th record and return the dict
        chosen_record = self.records[index]

        if self.preprocess_on_batch:
            chosen_record = [chosen_record]
            chosen_record = self._preprocess(chosen_record)
            chosen_record = chosen_record[0]
        assert type(chosen_record) == dict, "This should be a dict"

        # check that the rgbs and depths are in order
        assert chosen_record['rgb_camXs'].shape[1] == 3, "Channel dimension must be 1"
        assert chosen_record['rgb_camRs'].shape[1] == 3, "channel dim must be 1"
        assert chosen_record['depth_camXs'].shape[1] == 1, "channel dim must be 1"

        S = self.S
        nviews = len(chosen_record['rgb_camXs'])

        indices = random.sample(range(nviews), S)

        if self.fixed_view:
            indices = self.fixed_view_list[:S]

        item_names = [
                'rgb_camXs',
                'depth_camXs',
                'pix_T_cams',
                'origin_T_camXs',
                'origin_T_camRs',
                'xyz_camXs',
                'xyz_camRs',
                ]
        for item_name in item_names:
            assert item_name in list(chosen_record.keys())
            #print("item_name", item_name)
            #print(chosen_record[item_name].shape)
            #print("===", indices)
            chosen_record[item_name] = chosen_record[item_name][indices]
        chosen_record["indices"] = indices

        # more checks I know that rgbs are the same and so are origin_T_camRs
        # not sure what this is doing
        assert torch.all(torch.eq(chosen_record['origin_T_camRs'], chosen_record['origin_T_camRs'][0])).item()
        assert torch.all(torch.eq(chosen_record['rgb_camRs'], chosen_record['rgb_camRs'][0])).item()

        # now that all the basic checks are complete return the record
        return chosen_record

class MetricLearnerData(torch.utils.data.Dataset):
    def __init__(self, dataset_path, plot=False):
        with open(dataset_path, 'r') as f:
            all_files = f.readlines()

        records = [line.strip('\n') for line in all_files]
        nrecords = len(records)
        print(f'found {nrecords} in {dataset_path}')
        for record in records:
            assert os.path.isfile(record), f"Record {record} was not found"

        self.device = torch.device('cuda')
        self.records = self._preprocess(records)


        ## what did the _preprocess do
        # I have lets say 5 records and each record has 51 values for each key
        # Say my hyp.S = 10, I am breaking each record into 5 parts
        # so a total of 25 parts, so basically the size of the records should be 25

        self.plot = plot
        if self.plot:
            # create a temp directory to save images and depths
            self.temp_dir = "/projects/katefgroup/gauravp/touch_project/temp_im_data"
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            self.color_dir = f'{self.temp_dir}/colors'
            self.depth_dir = f'{self.temp_dir}/depths'
            if not os.path.exists(self.color_dir):
                os.makedirs(self.color_dir)
            if not os.path.exists(self.depth_dir):
                os.makedirs(self.depth_dir)

    def _preprocess(self, records, print=False):
        seq_records = list()
        label_dict = {}
        obj_id_dict = {} #dictionary to maintain unique object identifier
        self.idx_to_label = {}
        self.label_clusters = {}
        record_idx = 0
        label = 0
        for record in records:
            # I have checked this above but again doing it here
            if not os.path.exists(record):
                raise FileNotFoundError(f'{record} not found')

            ### Assumes that the record path looks like /projects/katefgroup/shared_quant_models/data_with_bbox/cups_data/visual_data_1d18255a04d22794e521eeb8bb14c5b3.npy
            # TODO PUT these two guys in npz file
            obj_type = record.split("/")[5]
            obj_name = record.split("/")[6]

            assert len(record.split("/")) == 7
            if obj_type not in label_dict:
                label = len(label_dict.keys())
                label_dict[obj_type] = label
            else:
                label = label_dict[obj_type]

            if obj_name not in obj_id_dict:
                object_id = len(obj_id_dict.keys())
                obj_id_dict[obj_name] = object_id
            else:
                #This shouldn't usually happen --- we shouldn't have duplicate objects
                object_id = obj_id_dict[obj_name]

            data = np.load(record, allow_pickle=True).item()
            rgb_camXs = data.rgb_camXs
            depth_camXs = data.depth_camXs
            rgb_camRs = data.rgb_camRs
            extrinsics = data.extrinsics
            intrinsics = data.intrinsics
            origin_T_camR = np.linalg.inv(data.camR_T_origin)
            # now ref_frame_bbox should also be added to the mix
            bbox_in_ref_cam = data.bbox_in_ref_cam

            """"
            rgbs = np.transpose(rgb_camXs, [0,3,1,2])
            depths = depth_camXs
            rgb_refs = np.transpose(rgb_camRs, [0,3,1,2])
            """
            # below I am breaking 51 images into S equal parts randomly
            sampler = BatchSampler(SubsetRandomSampler(range(len(rgb_camXs))), hyp.S, drop_last=True)
            for i, indices in enumerate(sampler):
                rgbs = np.transpose(rgb_camXs[indices], [0,3,1,2])
                depths = depth_camXs[indices]
                rgb_refs = np.transpose(rgb_camRs[indices], [0,3,1,2])
                origin_T_camXs = extrinsics[indices]
                pix_T_cam = intrinsics[indices]
                origin_T_camRs = np.reshape(origin_T_camR, [1, 4, 4])
                origin_T_camRs = np.tile(origin_T_camRs, [len(indices), 1, 1])

                # convert everything to torch and tensor and cuda
                rgbs = torch.from_numpy(rgbs).float()
                rgb_refs = torch.from_numpy(rgb_refs).float()
                depths = torch.from_numpy(depths).float().unsqueeze(1)
                pix_T_cam = torch.from_numpy(pix_T_cam).float()
                origin_T_camXs = torch.from_numpy(origin_T_camXs).float()
                origin_T_camRs = torch.from_numpy(origin_T_camRs).float()

                # get the pointclouds as well here
                V = hyp.V
                xyz_cams = torch.zeros((hyp.S, V, 3))
                xyz_camRs = torch.zeros((hyp.S, V, 3))

                for s in list(range(hyp.S)):
                    depth = torch.unsqueeze(depths[s], axis=0)
                    single_intrinsic = torch.unsqueeze(pix_T_cam[s], axis=0)
                    xyz_cam = utils.geom.depth2pointcloud(depth, single_intrinsic,
                            device=torch.device('cpu')).squeeze()

                    # depth clipping
                    # xyz_cam = xyz_cam[xyz_cam[:, 2] > 0.01]
                    # xyz_cam = xyz_cam[xyz_cam[:, 2] < 5.0]

                    assert len(xyz_cam) == V, "this should be V as no clipping is done"

                    if xyz_cam.shape[0] < V:
                        xyz_cam = torch.nn.functional.pad(xyz_cam, (0,0,0,V-xyz_cam.shape[0]),'constant',0)

                    xyz_cams[s] = xyz_cam
                    camR_T_camX = torch.mm(torch.inverse(origin_T_camRs[s]), origin_T_camXs[s])
                    xyz_camR = utils.geom.apply_4x4(camR_T_camX.unsqueeze(0), xyz_cam.unsqueeze(0)).squeeze(0)
                    xyz_camRs[s] = xyz_camR

                # fill all this up in a dictionary and add to list
                d = dict()
                rgbs = rgbs / 255.
                rgb_refs = rgb_refs / 255.
                rgbs = rgbs - 0.5
                rgb_refs = rgb_refs - 0.5

                d['rgb_camXs'] = rgbs
                d['rgb_camRs'] = rgb_refs
                d['depth_camXs'] = depths
                d['pix_T_cams'] = pix_T_cam
                d['origin_T_camXs'] = origin_T_camXs
                d['origin_T_camRs'] = origin_T_camRs
                d['xyz_camXs'] = xyz_cams
                d['xyz_camRs'] = xyz_camRs
                d['bbox_in_ref_cam'] = bbox_in_ref_cam
                d['record'] = record
                d['label'] = label
                d['object_id'] = object_id

                self.idx_to_label[record_idx] = label
                if label in self.label_clusters:
                    self.label_clusters[label].append(record_idx)
                else:
                    self.label_clusters[label] = [record_idx]
                record_idx += 1
                seq_records.append(d)

        return seq_records


    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # select the i-th record and return the dict
        chosen_record = self.get_single_item(index)
        pos_label = self.idx_to_label[index]
        per_bin_samples = (hyp.MB - 1) / (len(self.label_clusters.keys()) - 1)
        if per_bin_samples < 1:
            per_bin_samples = 1

        records = [chosen_record]
        label_keys = list(self.label_clusters.keys())
        random.shuffle(label_keys)
        # Find B - 1 negative samples for this positive sample
        for label in label_keys:
            if len(records) < hyp.MB and label != pos_label:
                bin_record_idx = random.choices(self.label_clusters[label], k=int(per_bin_samples))
                for idx in bin_record_idx:
                    records.append(self.get_single_item(idx))
        return records

    def get_single_item(self, index):
        # select the i-th record and return the dict
        chosen_record = self.records[index]
        assert type(chosen_record) == dict, "This should be a dict"

        # check that the rgbs and depths are in order
        assert chosen_record['rgb_camXs'].shape[1] == 3, "Channel dimension must be 1"
        assert chosen_record['rgb_camRs'].shape[1] == 3, "channel dim must be 1"
        assert chosen_record['depth_camXs'].shape[1] == 1, "channel dim must be 1"

        item_names = [
                'rgb_camXs',
                'rgb_camRs',
                'depth_camXs',
                'pix_T_cams',
                'origin_T_camXs',
                'origin_T_camRs',
                'xyz_camXs',
                'xyz_camRs',
                ]
        for i in item_names:
            assert i in list(chosen_record.keys())

        # more checks I know that rgbs are the same and so are origin_T_camRs
        assert torch.all(torch.eq(chosen_record['origin_T_camRs'], chosen_record['origin_T_camRs'][0])).item()
        assert torch.all(torch.eq(chosen_record['rgb_camRs'], chosen_record['rgb_camRs'][0])).item()
        return chosen_record

def metric_learner_collate(batch):
    #print("In metric learner collate")
    # We get a list of dicts, we need to convert it to dict of lists and return
    keys = batch[0].keys()
    records_dict = {}
    for record in batch:
        for k in keys:
            if k in records_dict:
                if type(record[k]) == list:
                    records_dict[k] =  records_dict[k] + record[k]
                else:
                    records_dict[k] = torch.cat((records_dict[k], record[k]))
            else:
                records_dict[k] = record[k]
    return records_dict

class NpzRecordDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path) as f:
            content = f.readlines()
        records = [hyp.dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))

        self.records = records

        # self.sess = tf.Session()

        # dataset = tf.data.TFRecordDataset(
        #     records,
        #     compression_type="GZIP"
        # ).repeat()

    def __getitem__(self, index):
        if hyp.dataset_name=='carla' or hyp.dataset_name=='kitti':
            filename = self.records[index]
            d = np.load(filename)

            # print(d['pix_T_cams'])

        else:
            assert(False) # reader not ready yet

        # batch_torch = []
        # for b in batch:
        #     batch_torch.append(torch.tensor(b))

        item_names = [
        'pix_T_cams',
        'cam_T_velos',
        'origin_T_camRs',
        'origin_T_camXs',
        'rgb_camRs',
        'rgb_camXs',
        'xyz_veloXs',
        'boxes3D',
        'tids',
        'scores',
        ]

        d = dict(d)
        #d_tensor = dict()

        if hyp.do_time_flip:
            d = random_time_flip_single(d)

        rgb_camRs = d['rgb_camRs']
        rgb_camXs = d['rgb_camXs']

        # move channel dim inward, like pytorch wants
        rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])
        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camRs = utils.improc.preprocess_color(rgb_camRs)
        rgb_camXs = utils.improc.preprocess_color(rgb_camXs)

        d['rgb_camRs'] = rgb_camRs
        d['rgb_camXs'] = rgb_camXs

        # for item_name in item_names:
        #     item = d[item_name]
        #     d_tensor[item_name] = torch.tensor(item)

        # d = {}
        # [d['pix_T_cams'],
        #  d['cam_T_velos'],
        #  d['origin_T_camRs'],
        #  d['origin_T_camXs'],
        #  d['rgb_camRs'],
        #  d['rgb_camXs'],
        #  d['xyz_veloXs'],
        #  d['boxes3D'],
        #  d['tids'],
        #  d['scores'],
        #  ] = batch_torch

        # print(d['rgb_camRs'].dtype)

        return d

    def __len__(self):
        return len(self.records)


def random_time_flip_batch(batch):
    pix_T_cams = batch['pix_T_cams']
    cam_T_velos = batch['cam_T_velos']
    origin_T_camRs = batch['origin_T_camRs']
    origin_T_camXs = batch['origin_T_camXs']
    #
    rgb_camRs = batch['rgb_camRs']
    rgb_camXs = batch['rgb_camXs']
    xyz_veloXs = batch['xyz_veloXs']
    #
    boxes3D = batch['boxes3D']
    tids = batch['tids']
    scores = batch['scores']

    # let's do this for the whole batch at once, for simplicity
    # do_flip = tf.cast(tf.random_uniform([1],minval=0,maxval=2,dtype=tf.int32), tf.bool)
    do_flip = torch.rand(1)

    item_names = [
        'pix_T_cams',
        'cam_T_velos',
        'origin_T_camRs',
        'origin_T_camXs',
        'rgb_camRs',
        'rgb_camXs',
        'xyz_veloXs',
        'boxes3D',
        'tids',
        'scores',
    ]
    for item_name in item_names:
        item = batch[item_name]
        if do_flip > 0.5:
            # flip along the seq dim
            item = item.flip(1)
        batch[item_name] = item

    return batch

def random_time_flip_single(batch):
    # pix_T_cams = batch['pix_T_cams']
    # cam_T_velos = batch['cam_T_velos']
    # origin_T_camRs = batch['origin_T_camRs']
    # origin_T_camXs = batch['origin_T_camXs']
    # #
    # rgb_camRs = batch['rgb_camRs']
    # rgb_camXs = batch['rgb_camXs']
    # xyz_veloXs = batch['xyz_veloXs']
    # #
    # boxes3D = batch['boxes3D']
    # tids = batch['tids']
    # scores = batch['scores']

    # let's do this for the whole batch at once, for simplicity
    # do_flip = tf.cast(tf.random_uniform([1],minval=0,maxval=2,dtype=tf.int32), tf.bool)
    do_flip = torch.rand(1)

    item_names = [
        'pix_T_cams',
        'cam_T_velos',
        'origin_T_camRs',
        'origin_T_camXs',
        'rgb_camRs',
        'rgb_camXs',
        'xyz_veloXs',
        'boxes3D',
        'tids',
        'scores',
    ]
    for item_name in item_names:
        item = batch[item_name]
        if do_flip > 0.5:
            if torch.is_tensor(item):
                # flip along the seq dim
                item = item.flip(0)
            else: #numpy array
                item = np.flip(item, axis=0)
        batch[item_name] = item

    return batch

def get_inputs(config):
    dataset_format = config.dataset_format
    all_set_inputs = {}


    data_root_dir = utils.utils.get_data_dir()
    for set_name in config.data_paths:
        data_path = config.data_paths[set_name]
        if not data_path.startswith("/"):
            data_path = os.path.join(data_root_dir, data_path)
            config.data_paths[set_name] = data_path

    for set_name in config.set_names:
        if config.sets_to_run[set_name]:
            data_path = config.data_paths[set_name]  # this will be a txt file for me

            shuffle = config.shuffles[set_name]
            if dataset_format == 'tf':
                all_set_inputs[set_name] = TFRecordDataset(dataset_path = data_path, shuffle=shuffle)
            elif dataset_format == 'npz':
                all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset(dataset_path = data_path), \
                shuffle=shuffle, batch_size=config.B, num_workers=4, pin_memory=True)
            elif dataset_format == 'txt':


                if config.do_metric_learning:
                    print('---- I am loading data for metric learning + mujoco offline -----')
                    all_set_inputs[set_name] = torch.utils.data.DataLoader(
                        dataset=MetricLearnerData(dataset_path=data_path, plot=False),
                        shuffle=shuffle,
                        batch_size=config.B,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True
                    )
                elif config.do_mujoco_offline or config.do_mujoco_offline_metric or config.do_mujoco_offline_metric_2d:
                    print('---- I am loading mujoco offline data ----')
                    if set_name == 'train':
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(
                            dataset=MuJoCoOfflineData(config, dataset_path=data_path, plot=False, num_workers=1, preprocess_on_batch=True),
                            shuffle=shuffle,
                            batch_size=config.B,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=True
                        )

                    elif set_name == 'val' or set_name == 'test':
                        print('------- Valset is being created, train is file --------')
                        val_dataset = MuJoCoOfflineData(config, dataset_path=data_path,
                            plot=False, train=False)
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(
                            val_dataset, shuffle=False, batch_size=config.B,
                            num_workers=1, pin_memory=True, drop_last=True
                        )
                    else:
                        raise ValueError
                elif config.do_touch_embed:
                    print('----  I am loading touch embed data ----')
                    all_set_inputs[set_name] = torch.utils.data.DataLoader(
                        dataset=TouchEmbedData(dataset_path=data_path, plot=False,
                                               set_name=set_name),
                        shuffle=shuffle,
                        batch_size=config.B,
                        # num_workers=4,
                        pin_memory=True,
                        drop_last=True
                    )
            else:
                assert False #what is the data format?

    return all_set_inputs
