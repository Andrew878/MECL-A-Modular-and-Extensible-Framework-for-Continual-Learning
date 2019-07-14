
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import max, zeros
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy



class DataSetAndInterface:

    def __init__(self, name, dataset, path, transformations, original_channel_number,original_input_dimensions):

        self.name = name
        self.dataset = dataset
        self.path = path
        self.categories_list = dataset['train']['VAE'].classes

        # The EMNIST class labels in the torchvision module are incorrect. This hack fixes it.
        if(name == 'EMNIST'):
            self.categories_list = [chr(i) for i in range(ord('a'), ord('z')+1)]

        self.num_categories = len(self.categories_list)
        self.transformations = transformations
        self.original_channel_number = original_channel_number
        self.original_input_dimensions = original_input_dimensions
        self.training_set_size = len(dataset['train']['VAE'])
        print(self.training_set_size )
        self.val_set_size = len(dataset['val']['VAE'])
        print(self.val_set_size )
        self.dataset_splits = ['train', 'val']



    def return_data_loaders(self, branch_component, dataset_selected = None, BATCH_SIZE = 64, num_workers=4):

        if (dataset_selected == None):
            dataset_selected = self.dataset

        dataloaders = {}
        for d_set in self.dataset_splits:
            dataloaders[d_set] = torch.utils.data.DataLoader(dataset_selected[d_set][branch_component], batch_size=BATCH_SIZE,shuffle=True, num_workers=num_workers)


        return dataloaders

    def obtain_dataset_with_subset_of_categories(self, branch_component, category_subset):

        """https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276"""

        subset_dataset = copy.deepcopy(self.dataset)

        for d_set in self.dataset_splits:
            idx = subset_dataset.targets[d_set][branch_component] in category_subset
            subset_dataset.targets = subset_dataset.targets[idx]
            subset_dataset.data = subset_dataset.data[idx]

        return subset_dataset

    def obtain_dataloader_with_subset_of_categories(self, branch_component, category_subset, BATCH_SIZE = 64, num_workers=4):
        return self.return_data_loaders(branch_component,self.obtain_dataset_with_subset_of_categories(branch_component,category_subset),BATCH_SIZE,num_workers)

