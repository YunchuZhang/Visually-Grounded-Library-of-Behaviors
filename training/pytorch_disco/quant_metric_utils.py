import os
import collections
import numpy as np

NUMPY_RANDOM = np.random

# blatantly copied from pytorch-metric-learning repo
def get_labels_to_indices(labels):
    labels_dict = collections.defaultdict(list)
    for i, l in enumerate(labels):
        labels_dict[l].append(i)

    # now convert each value to numpy array
    for k, v in labels_dict.items():
        labels_dict[k] = np.array(v, dtype=np.int)

    return labels_dict

def safe_random_choice(input_data, size):
    """
    Randomly samples without replacement from a sequence. It is "safe" because
    if len(input_data) < size, it will randomly sample WITH replacement
    Args:
        input_data is a sequence, like a torch tensor, numpy array,
                        python list, tuple etc
        size is the number of elements to randomly sample from input_data
    Returns:
        An array of size "size", randomly sampled from input_data
    """
    replace = len(input_data) < size
    return NUMPY_RANDOM.choice(input_data, size=size, replace=replace)
