import os
import torch
import random
import numpy as np
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

def make_txt_file(data_folder, mode):
    if not os.path.exists(data_folder):
        raise FileNotFoundError
    
    dir_path = "quant_train_files"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = f"{dir_path}/{mode}_file.txt"
    with open(file_path, "w") as f:
        for fw in os.listdir(data_folder):
            f.write(f'{os.path.join(data_folder, fw)}\n')
    f.close()

class ValidationLoader(object):
    def __init__(self, dataset_txt):
        if not os.path.exists(dataset_txt):
            raise FileNotFoundError

        with open(dataset_txt, 'r') as f:
            lines = f.readlines()
        f.close()

        self.lines = [l.strip('\n') for l in lines]
        self.records = self._preprocess()
    
    def _preprocess(self):
        records = list()
        for l in self.lines:
            assert os.path.exists(l)
            npz_record = np.load(l, allow_pickle=True).item()

            save_dict = dict(
                ob_tensor = torch.from_numpy(npz_record['tensor']).float(),
                label = npz_record['class_label'],
                file_name = l,
                ob_name = npz_record['ob_name'],
                class_name = npz_record['class_name']
            )
            records.append(save_dict)
        return records
    
    def __len__(self):
        return len(self.records)
    
    def yield_data(self):
        return iter(self.records)

class BaseDataset(Dataset):
    def __init__(self, dataset_txt, transforms=None):
        if not os.path.exists(dataset_txt):
            raise FileNotFoundError

        self.dataset_txt = dataset_txt
        return_tuple = self.preprocess_()
        self.data_files = return_tuple[0]
        self.tensors = return_tuple[1]
        self.targets = return_tuple[2]
        self.class_names = return_tuple[3]
        self.object_names = return_tuple[4]

        self.tensors = np.stack(self.tensors, axis=0)
        self.targets = np.asarray(self.targets).astype(int)

    def preprocess_(self):
        with open(self.dataset_txt, 'r') as f:
            lines = f.readlines()
        f.close()
        data_files = [d.strip('\n') for d in lines]
        # each is a npy file which contains a dict
        objects = list()
        targets = list()
        tensors = list()
        class_names = list()
        for d in data_files:
            assert os.path.exists(d), "npy files needed containing data"
            npy_data = np.load(d, allow_pickle=True).item()
            tensors.append(npy_data['tensor'].squeeze(0))
            targets.append(npy_data['class_label'])
            objects.append(npy_data['ob_name'])
            class_names.append(npy_data['class_name'])

        return data_files, tensors, targets, class_names, objects

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        tensor3D, label = self.tensors[idx], self.targets[idx]
        return tensor3D, label


class MetricLearningData(Dataset):
    def __init__(self, dataset_txt, transforms=None):
        # data_dir: directory containing extracted object tensors
        if not os.path.exists(dataset_txt):
            raise FileNotFoundError('please provide directory containing object tensors')
        
        self.dataset_txt = dataset_txt
        self.class_labels, self.class_data = self.preprocess_()
        self.class_labels = list(self.class_labels)
    
    def preprocess_(self):
        with open(self.dataset_txt, 'r') as f:
            lines = f.readlines()
        f.close()
        data = [d.strip('\n') for d in lines]
        
        # now separate the data into bins based on the number of classes
        unique_labels = set()
        per_class_data_files = defaultdict(list)
        for d in data:
            # load the file and check which class it belongs to
            assert os.path.exists(d), "expected a valid npz file with metric learning data"
            npz_data = np.load(d, allow_pickle=True).item()
            unique_labels.add(npz_data['class_label'])
            per_class_data_files[npz_data['class_label']].append(d)

        return unique_labels, per_class_data_files
    
    def __len__(self):
        return sum([len(self.class_data[d]) for d in self.class_data.keys()])
    
    def __getitem__(self, idx):
        # get the item first, it is file path, I am ignoring the idx completely is it a good thing
        # to do #TODO: ask people about it
        
        # step 1 : choose one the classes randomly
        chosen_class_label = random.choice(self.class_labels)
        # choose file from the class
        chosen_class_file = random.choice(self.class_data[chosen_class_label])
        
        # step 2: Choose a negative class label, this could be anything except the one above
        while True:
            neg_class_label = random.choice(self.class_labels)
            if not (neg_class_label == chosen_class_label): break
        neg_class_file = random.choice(self.class_data[neg_class_label])

        # Step 3: load the two files and get the tensors and class labels and return
        chosen_file_data = np.load(chosen_class_file, allow_pickle=True).item()
        neg_file_data = np.load(neg_class_file, allow_pickle=True).item()
        
        return_dict = dict(pos_tensor=chosen_file_data['tensor'].squeeze(0),
                           pos_label=chosen_file_data['class_label'],
                           neg_tensor=neg_file_data['tensor'].squeeze(0),
                           neg_label=neg_file_data['class_label'],
                           pos_file=chosen_class_file,
                           neg_file=neg_class_file)
            
        return return_dict
    
if __name__ == '__main__':
    # data_folder = "/home/ubuntu/pytorch_disco/extracted_tensor/02_m64x64x64_p64x64_1e-3_F32_Oc_c1_s1_Ve_d32_c1_train_file_controller_a01/01_m64x64x64_p64x64_F32f_Ocf_Vef_d32_c1_val_file_controller"
    # make_txt_file(data_folder, mode="val")
    from metric_data_samplers import MPerClassSampler

    # m_dataset = MetricLearningData(dataset_txt="/home/ubuntu/pytorch_disco/backend/quant_train_files/train_file.txt")
    # how should this be tested ideally
    # print(f'number of training 3d tensors are {len(m_dataset)}')
    # t_dataloader = torch.utils.data.DataLoader(m_dataset, batch_size=4, shuffle=True, drop_last=True)
    # import ipdb; ipdb.set_trace()

    dataset_txt = '/Users/gspat/aws_results_Dir/pytorch_disco/backend/quant_train_files/train_files_dt1542020.txt'
    train_dataset = BaseDataset(dataset_txt, transforms=None)
    sampler = MPerClassSampler(labels=train_dataset.targets, m=8,
                               length_before_new_iter=len(train_dataset))

    # now form the data-loader using sampler and dataset object
    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler,
                                  drop_last=True, collate_fn=None, shuffle=sampler is None,
                                  pin_memory=False)
    train_dataloader_iter = iter(train_dataloader)
    # now iterate through the dataset, using the method in pytorch-metric-learning
    start_epoch = 1
    max_epochs = 2
    for _ in range(start_epoch, max_epochs+1):
        try:
            train_dataloader_iter, curr_batch = train_dataloader_iter, next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            curr_batch = next(train_dataloader_iter)

        from IPython import embed; embed()
