
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import max, zeros
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



class DataSetAndInterface:

    def __init__(self, name, dataset, path, transformations, original_channel_number,original_input_dimensions):

        self.name = name
        self.dataset = dataset
        self.path = path
        self.categories_list = dataset['train']['VAE'].classes
        self.num_categories = len(self.categories_list)
        self.transformations = transformations
        self.original_channel_number = original_channel_number
        self.original_input_dimensions = original_input_dimensions
        self.training_set_size = len(dataset['train']['VAE'])
        print(self.training_set_size )
        self.val_set_size = len(dataset['val']['VAE'])
        print(self.val_set_size )



    def return_data_loaders(self, branch_component, BATCH_SIZE = 64, num_workers=4):

        dataloaders = {}
        for d_set in ['train', 'val']:

            dataloaders[d_set] = torch.utils.data.DataLoader(self.dataset[d_set][branch_component], batch_size=BATCH_SIZE,shuffle=True, num_workers=num_workers)

        return dataloaders


