import torch
import torch.optim
import CVAE
import Utils
import time
import copy
from torchvision import models
import torch.nn as nn


class Gate:

    """The Gate is responsible for managing the various task branches. Is used for allocating samples to a branch, and measuring task relatedness"""

    def __init__(self):
        self.task_branch_dictionary = {}

    def add_task_branch(self, *new_tasks):
        for task_branch in new_tasks:
            self.task_branch_dictionary[task_branch.task_name] = task_branch

    def replace_task_branch(self, old, new):
        del self.task_branch_dictionary[old.task_name]
        self.task_branch_dictionary[new.task_name] = new

    def allocate_sample_to_task_branch(self, x, is_standardised_distance_check=True, is_return_both_metrics=False):

        """Takes an unseen sample, and allocates to appropriate task branch. Can allocate by using absolute or relative standard distance measure"""

        # initialisation
        lowest_std_dev_best_class = None
        lowest_recon_error_best_class = None
        lowest_std_dev_distance = float('inf')
        lowest_recon_error = float('inf')
        lowest_recon_error_best_task_branch = None
        lowest_std_dev_distance_best_task_branch = None

        # cycle through each branch, find lowest reconstruction error within all branches
        for task_branch in self.task_branch_dictionary.values():

            # find lowest reconstruction error within chosen branch
            results_information, recon_x = task_branch.given_observation_find_lowest_reconstruction_error(x,
                                                                                                 is_standardised_distance_check)
            # check if best result among all branches for absolute measure
            if results_information[0][1] < lowest_recon_error:
                lowest_recon_error = results_information[0][1]
                lowest_recon_error_best_class = results_information[0][0]
                lowest_recon_error_best_task_branch = task_branch

            # check if best result among all branches for relative measure
            if is_standardised_distance_check:
                if results_information[1][1] < lowest_std_dev_distance:
                    lowest_std_dev_distance = results_information[1][1]
                    lowest_std_dev_best_class = results_information[1][0]
                    lowest_std_dev_distance_best_task_branch = task_branch

        # return the best task, and sample reconstruction error (depending if absolute or relative chosen or both)
        if is_return_both_metrics and is_standardised_distance_check:
            return lowest_recon_error_best_task_branch, lowest_recon_error_best_class, lowest_std_dev_distance_best_task_branch, lowest_std_dev_best_class, recon_x

        if is_standardised_distance_check:
            return lowest_std_dev_distance_best_task_branch, lowest_std_dev_best_class, recon_x
        else:
            return lowest_recon_error_best_task_branch, lowest_recon_error_best_class, recon_x

    def given_new_dataset_find_best_fit_domain_from_existing_tasks(self, datasetAndinterface, category_subset,
                                                                   num_samples_to_check=100):

        """Takes any dataset (either full dataset or subset of categories), and finds the best fit from pre-trained, existing tasks. Used for the purpose of identifying which pretrained model to use for transfer learning for a new dataset"""

        print("*** Checking the following new data: ", datasetAndinterface.name, "Size: ",num_samples_to_check)

        # if no subset is provided,use all categories
        if len(category_subset) == 0:
            dataloader = datasetAndinterface.return_data_loaders(branch_component='VAE', BATCH_SIZE=1)['val']
        else:
            dataloader = datasetAndinterface.obtain_dataloader_with_subset_of_categories(branch_component='VAE',
                                                                                         split='val',
                                                                                         category_subset=category_subset,
                                                                                         BATCH_SIZE=1)

        # Cycle through each task and find the task with highest task relatedness on sample
        most_related_task = None
        best_task_relativity = 0
        for task_branch in self.task_branch_dictionary.values():
            reconstruction_error_average, task_relatedness = task_branch.given_task_data_set_find_task_relatedness(
                dataloader, num_samples_to_check=num_samples_to_check)
            print("   --- Checking against:", task_branch.task_name)
            print("   --- Reconstruction error average", reconstruction_error_average, " Task relatedness",
                  task_relatedness)
            if (task_relatedness > best_task_relativity):
                best_task_relativity = task_relatedness
                most_related_task = task_branch

        print("Closest task is ", most_related_task.task_name, "with a task relativity of", best_task_relativity)


    def classify_input_using_allocation_method(self, x):

        "Take an unseen observation, allocate to branch and then classify"

        # find best task
        lowest_recon_error_best_task_branch, lowest_recon_error_best_class, recon_x = self.allocate_sample_to_task_branch(x,
                                                                                                                 is_standardised_distance_check=False,
                                                                                                                  is_return_both_metrics=False)
        # convert to CNN dimensions
        x = lowest_recon_error_best_task_branch.dataset_interface.transformations['CNN']['test_to_image'](torch.squeeze(x))
        x = torch.unsqueeze(x, 0)

        # make prediction using expert model
        pred_cat, probability = lowest_recon_error_best_task_branch.classify_with_CNN(x)

        return lowest_recon_error_best_task_branch, pred_cat, probability
