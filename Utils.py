import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import max, zeros
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def idx2onehot(idx, num_categories):
    """Returns a one_hot_encoded_vector Borrowed from xxxx"""

    assert idx.shape[1] == 1
    # print( n)
    assert torch.max(idx).item() < num_categories

    onehot = torch.zeros(idx.size(0), num_categories)
    onehot.scatter_(1, idx.data, 1)

    return onehot


def generate_list_of_class_category_tensors(N_CLASSES):
    class_list = []
    for j in range(0, N_CLASSES):
        class_list.append(torch.tensor(np.array([[j], ])).to(dtype=torch.long))
    return class_list


def test_generating_and classification_ability(self, task_branches):
    for task_branch in task_branches:
        print("Task: ", task_branch.task_name)

        synthetic_data_list_x, synthetic_data_list_y = task_branch.VAE_most_recent.generate_synthetic_set_all_cats()
        task_branch.classify_with_CNN()
