import torch
from torch.utils.data.sampler import Sampler
import quant_metric_utils as c_f

# borrowed from pytorch-metric-learning
class MPerClassSampler(Sampler):
    def __init__(self, labels, m, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        self.m_per_class = int(m)
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.length_of_single_pass < self.list_size:
            self.list_size -= (self.list_size % self.length_of_single_pass)

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.list_size // self.length_of_single_pass if self.length_of_single_pass < self.list_size else 1
        for _ in range(num_iters):
            c_f.NUMPY_RANDOM.shuffle(self.labels)
            for label in self.labels:
                t = self.labels_to_indices[label]
                idx_list[i:i + self.m_per_class] = c_f.safe_random_choice(t, size=self.m_per_class)
                i += self.m_per_class
        return iter(idx_list)
