
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

        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        self.transformations = transformations
        self.original_channel_number = original_channel_number
        self.original_input_dimensions = original_input_dimensions
        self.training_set_size = len(dataset['train']['VAE'])
        print(self.training_set_size )
        self.val_set_size = len(dataset['val']['VAE'])
        print(self.val_set_size )
        self.dataset_splits = ['train', 'val']
        self.mutated_count = 0

    def update_data_set(self, dataset, is_synthetic_change = True):
        if is_synthetic_change:
            self.mutated_count += 1
        self.dataset = dataset
        self.categories_list = dataset['train']['VAE'].classes
        print("cat list",self.categories_list)
        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        print("num cats",self.num_categories)
        self.training_set_size = len(dataset['train']['VAE'])
        print(self.training_set_size)
        self.val_set_size = len(dataset['val']['VAE'])
        print(self.val_set_size)



    def return_data_loaders(self, branch_component, dataset_selected = None, BATCH_SIZE = 64, num_workers=0):



        if (dataset_selected == None):
            dataset_selected = self.dataset

        dataloaders = {}
        for d_set in self.dataset_splits:
            dataloaders[d_set] = torch.utils.data.DataLoader(dataset_selected[d_set][branch_component], batch_size=BATCH_SIZE,shuffle=True, num_workers=num_workers)

            # there is a dimension discrepancy between what resnet requires and what the VAE outputs. This fixes it
            # in the CustomDataSetAndLoader class
            if self.mutated_count > 0 and branch_component == 'CNN':
                dataset_selected[d_set][branch_component].is_make_dim_adjustment_for_resnet = True

        return dataloaders

    def obtain_dataset_with_subset_of_categories(self, branch_component, category_subset):


        """https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276"""

        #print(self.label_to_index_dict)
        #print(category_subset)
        category_subset_indices = [self.label_to_index_dict[i] for i in category_subset]
     #   print(category_subset_indices)


        subset_dataset = copy.deepcopy(self.dataset)

        for d_set in self.dataset_splits:
            idx = torch.zeros(len(subset_dataset[d_set][branch_component]),dtype=torch.uint8)
            #print(d_set)
            #print(idx)
            for cat_index in category_subset_indices:
                #print(cat_index)
                #print(subset_dataset[d_set][branch_component].targets)
                id_cat = (subset_dataset[d_set][branch_component].targets == cat_index)
                #print("id_cat",id_cat)
                #print("idx",id_cat)
                idx += id_cat
                #print("idx",idx)


            subset_dataset[d_set][branch_component].targets = subset_dataset[d_set][branch_component].targets[idx]
            subset_dataset[d_set][branch_component].data = subset_dataset[d_set][branch_component].data[idx]
            #print("len(subset_dataset[d_set][branch_component].data)",len(subset_dataset[d_set][branch_component].data))

        return subset_dataset

    def obtain_dataloader_with_subset_of_categories(self, branch_component, category_subset, BATCH_SIZE = 64, num_workers=4):


        return self.return_data_loaders(branch_component,self.obtain_dataset_with_subset_of_categories(branch_component,category_subset),BATCH_SIZE,num_workers)


    def add_outside_data_to_data_set(self, datasetAndInterfaceTwo, sub_set_list_cats_two=[]):

        print("num cats", self.num_categories)
        self.training_set_size = len(self.dataset['train']['VAE'])
        #print(self.training_set_size)
        self.val_set_size = len(self.dataset['val']['VAE'])
        #print(self.val_set_size)



        # if specific labels aren't given, then add all
        if len(sub_set_list_cats_two) == 0:
            sub_set_list_cats_two = datasetAndInterfaceTwo.categories_list

        for model in ['VAE', 'CNN']:

            subset_dataset_two = datasetAndInterfaceTwo.obtain_dataset_with_subset_of_categories(model, sub_set_list_cats_two)

            for split in ['train', 'val']:
                freq_check = {i: 0 for i in range(0, 30)}
                #print(subset_dataset_two[split][model].targets)
                #print("len(subset_dataset_two[split][model])",len(subset_dataset_two[split][model]))
                subset_dataset_two[split][model].targets = subset_dataset_two[split][model].targets + self.num_categories
                #print(subset_dataset_two[split][model].targets)
                print("REAL ONLY model", model, "split", split, "len(self.dataset[split][model])", len(self.dataset[split][model]),"len(subset_dataset_two[split][model])",len(subset_dataset_two[split][model]) )
                self.dataset[split][model] = self.dataset[split][model] + subset_dataset_two[split][model]
                print("REAL ONLY model", model, "split", split, "len(self.dataset[split][model])", len(self.dataset[split][model]))
                #print("real combined database targets", self.dataset[split][model].targets)
                #self.dataset[split][model] = torch.utils.data.ConcatDataset([self.dataset[split][model], datasetAndInterfaceTwo.dataset[split][model]])
                #print(self.dataset[split][model])

                for i in range(0,len(self.dataset[split][model])):
                    image, cat = self.dataset[split][model][i]
                    freq_check[cat] += 1
                print(freq_check)

        self.categories_list.extend(sub_set_list_cats_two)
        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        print("num cats", self.num_categories)
        self.training_set_size = len(self.dataset['train']['VAE'])
        #print(self.training_set_size)
        self.val_set_size = len(self.dataset['val']['VAE'])
        #print(self.val_set_size)
