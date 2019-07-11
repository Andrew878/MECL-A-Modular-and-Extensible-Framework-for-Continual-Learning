import torch
import torch.optim
import CVAE
import Utils
import time
import copy
from torchvision import models
import torch.nn as nn


class Gate:

    def __init__(self):
        self.task_branch_dictionary = {}


    def add_task_branch(self, *new_tasks):
        for task_branch in new_tasks:
            self.add_task_branch(task_branch)

    def replace_task_branch(self, old, new):
        del self.task_branch_dictionary[old.task_name]
        self.task_branch_dictionary[new.task_name] = new

    def allocate_sample_to_task_branch(self, x):
