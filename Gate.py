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
            self.task_branch_dictionary[task_branch.task_name] = task_branch

    def replace_task_branch(self, old, new):
        del self.task_branch_dictionary[old.task_name]
        self.task_branch_dictionary[new.task_name] = new

    def allocate_sample_to_task_branch(self, x, is_standardised_distance_check = True, is_return_both_metrics = False):

        lowest_std_dev_best_class = None
        lowest_recon_error_best_class  = None
        lowest_std_dev_distance = 10000000000
        lowest_recon_error = 10000000000

        lowest_recon_error_best_task_branch  = None
        lowest_std_dev_distance_best_task_branch  = None

        for task_branch in self.task_branch_dictionary.values():

            results_information = task_branch.given_observation_find_lowest_reconstruction_error(x, is_standardised_distance_check)

            if results_information[0][1] < lowest_recon_error:
                lowest_recon_error = results_information[0][1]
                lowest_recon_error_best_class = results_information[0][0]
                lowest_recon_error_best_task_branch = task_branch

            #print("Task ", task_branch.task_name,results_information)

            if is_standardised_distance_check:

                if results_information[1][1] < lowest_std_dev_distance:
                    lowest_std_dev_distance = results_information[1][1]
                    lowest_std_dev_best_class = results_information[1][0]
                    lowest_std_dev_distance_best_task_branch = task_branch


        if is_return_both_metrics and is_standardised_distance_check :
            #print(lowest_recon_error_best_task_branch.task_name, lowest_recon_error_best_class,lowest_std_dev_distance_best_task_branch.task_name, lowest_std_dev_best_class)
            return lowest_recon_error_best_task_branch, lowest_recon_error_best_class,lowest_std_dev_distance_best_task_branch, lowest_std_dev_best_class

        if is_standardised_distance_check:
            return lowest_std_dev_distance_best_task_branch, lowest_std_dev_best_class
        else:
            return lowest_recon_error_best_task_branch, lowest_recon_error_best_class
