
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import max, zeros
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import copy
import matplotlib.pyplot as plt


class DataSetAndInterface:

    def __init__(self, name, dataset, path, transformations, original_channel_number,original_input_dimensions, list_of_fixed_noise):

        self.name = name
        self.dataset = dataset
        self.dataset_original = copy.deepcopy(dataset)
        self.path = path
        self.list_of_fixed_noise = list_of_fixed_noise
        self.categories_list = dataset['train']['VAE'].classes
        self.transformations = transformations
        self.dataset_splits = ['train', 'val']#,'test']

        # The EMNIST class labels in the torchvision module are incorrect. This hack fixes it.
        if(name == 'EMNIST'):
            self.categories_list = [chr(i) for i in range(ord('a'), ord('z')+1)]

        # self.dataset_targets = {}
        # for split in self.dataset_splits:
        #     self.dataset_targets[split] = {}
        #     for model in ['VAE','CNN']:
        #         list_of_cats = []
        #         for image, cat in self.dataset[split][model]:
        #             list_of_cats.append(cat)
        #         self.dataset_targets[split][model] = torch.IntTensor(list_of_cats)
        #         print(name,split,model, self.dataset_targets[split][model])

        self.show_plots_of_dataset()

        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        self.original_channel_number = original_channel_number
        self.original_input_dimensions = original_input_dimensions
        self.training_set_size = len(dataset['train']['VAE'])
        print(self.training_set_size )
        self.val_set_size = len(dataset['val']['VAE'])
        print(self.val_set_size )
        self.mutated_count = 0

    def show_plots_of_dataset(self, is_random = True):
        fig1 = plt.figure(figsize=(15, 15))
        x = 0
        r = 3
        c = 3
        for i in range(x, r * c):
            img, cat = self.dataset['val']['VAE'][i]
            img = img.view(28, 28).data
            img = img.numpy()
            ax = fig1.add_subplot(r, c, i - x + 1)
            ax.axis('off')
            ax.set_title(cat)
            ax.imshow(img, cmap='gray_r')
        plt.ioff()
        plt.show()

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

    def reset_variables_to_initial_state(self):
        self.dataset = copy.deepcopy(self.dataset_original)
        self.categories_list = self.dataset['train']['VAE'].classes
        self.dataset_splits = ['train', 'val']#, 'test']

        # The EMNIST class labels in the torchvision module are incorrect. This hack fixes it.
        if (self.name == 'EMNIST'):
            self.categories_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]

        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        self.training_set_size = len(self.dataset['train']['VAE'])
        print(self.training_set_size)
        self.val_set_size = len(self.dataset['val']['VAE'])
        print(self.val_set_size)
        self.mutated_count = 0

    def return_data_loaders(self, branch_component, dataset_selected = None, BATCH_SIZE = 64, num_workers=0,split =None):



        if (dataset_selected == None):
            dataset_selected = self.dataset

        if (split == None):
            dataset_splits = self.dataset_splits
        else:
            dataset_splits = [split]
            data_loader = torch.utils.data.DataLoader(dataset_selected, batch_size=BATCH_SIZE,shuffle=True, num_workers=num_workers)
            return data_loader

        dataloaders = {}
        for d_set in dataset_splits:

            dataloaders[d_set] = torch.utils.data.DataLoader(dataset_selected[d_set][branch_component], batch_size=BATCH_SIZE,shuffle=True, num_workers=num_workers)

            # there is a dimension discrepancy between what resnet requires and what the VAE outputs. This fixes it
            # in the CustomDataSetAndLoader class
            if self.mutated_count > 0 and branch_component == 'CNN':
                dataset_selected[d_set][branch_component].is_make_dim_adjustment_for_resnet = True

        return dataloaders

    def obtain_dataset_with_subset_of_categories(self, branch_component, split, category_subset):

        """https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276"""

        is_count_freq = True
        is_plot_output = True

        print(self.label_to_index_dict)
        print(category_subset)
        category_subset_indices = [self.label_to_index_dict[i] for i in category_subset]
        print(category_subset_indices)


        subset_dataset = copy.deepcopy(self.dataset[split][branch_component])
        #subset_dataset_targets = copy.deepcopy(self.dataset_targets[split][branch_component])

        print("**********",branch_component, split)
        #print(idx)

        if is_count_freq:
            freq_check = {i: 0 for i in range(0, 26)}
            print("length subset_before",len(subset_dataset))
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1
            print(freq_check)

        idx = torch.zeros(len(subset_dataset),dtype=torch.uint8)
        for cat_index in category_subset_indices:
            # print(cat_index)
            print(subset_dataset.targets)
            id_cat = subset_dataset.targets == cat_index
            # print("id_cat",id_cat)
            # print("idx",id_cat)
            idx += id_cat
            # print("idx",idx)
        subset_dataset.targets = subset_dataset.targets[idx]
        subset_dataset.data = subset_dataset.data[idx]

        if is_count_freq:
            freq_check = {i: 0 for i in range(0, 26)}
            print("length subset_after", len(subset_dataset))
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1
            print(freq_check)

        if is_plot_output and branch_component == 'VAE':
            fig2 = plt.figure(figsize=(20, 20))
            x = 0
            r = 60
            c = 5

            print("TRYING TO PRINT Z'S")
            print(len(subset_dataset))
            for i in range(x, r * c):
                img, cat = subset_dataset[i]
                img = img.view(28, 28).data
                img = img.numpy()
                ax = fig2.add_subplot(r, c, i - x + 1)
                ax.axis('off')
                ax.set_title(cat)
                ax.imshow(img, cmap='gray_r')

            plt.ioff()
            plt.show()

        # print(self.label_to_index_dict)
        # print(category_subset)
        # category_subset_indices = [self.label_to_index_dict[i] for i in category_subset]
        # print(category_subset_indices)
        #
        # subset_dataset = copy.deepcopy(self.dataset)
        #
        # for d_set in self.dataset_splits:
        #     idx = torch.zeros(len(subset_dataset[d_set][branch_component]), dtype=torch.uint8)
        #     print(d_set)
        #     print(idx)
        #     for cat_index in category_subset_indices:
        #         print(cat_index)
        #         print(subset_dataset[d_set][branch_component].targets)
        #         id_cat = (subset_dataset[d_set][branch_component].targets == cat_index)
        #         print("id_cat", id_cat)
        #         print("idx", id_cat)
        #         idx += id_cat
        #         print("idx", idx)
        #
        #         subset_dataset[d_set][branch_component].targets = subset_dataset[d_set][branch_component].targets[idx]
        #         subset_dataset[d_set][branch_component].data = subset_dataset[d_set][branch_component].data[idx]
        #         print("len(subset_dataset[d_set][branch_component].data)",
        #               len(subset_dataset[d_set][branch_component].data))
        #
        # fig2 = plt.figure(figsize=(10, 10))
        # x = 0000
        # r = 2
        # c = 2

        return subset_dataset

    def obtain_dataloader_with_subset_of_categories(self, branch_component, split, category_subset, BATCH_SIZE = 64, num_workers=4):


        return self.return_data_loaders(branch_component,self.obtain_dataset_with_subset_of_categories(branch_component,split, category_subset),BATCH_SIZE,num_workers,category_subset)


    def add_outside_data_to_data_set(self, datasetAndInterfaceTwo, sub_set_list_cats_two=[]):

        #sub_set_list_cats_two = ['z','y']
        is_count_freq = True
        is_plot_true = False

        print("Real to Real num cats", self.num_categories)
        self.training_set_size = len(self.dataset['train']['VAE'])
        #print(self.training_set_size)
        self.val_set_size = len(self.dataset['val']['VAE'])
        #print(self.val_set_size)

        if is_count_freq:
            subset_dataset = self.dataset['train']['VAE']
            freq_check = {i: 0 for i in range(0, 26)}
            print("length subset_after", len(subset_dataset))
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1
            print(freq_check)


        # if specific labels aren't given, then add all
        if len(sub_set_list_cats_two) == 0:
            sub_set_list_cats_two = datasetAndInterfaceTwo.categories_list

        for model in ['VAE', 'CNN']:

            print("for ",model)

            for split in self.dataset_splits:
                subset_dataset_two = datasetAndInterfaceTwo.obtain_dataset_with_subset_of_categories(model, split, sub_set_list_cats_two)


                #print(subset_dataset_two[split][model].targets)
                #print("len(subset_dataset_two[split][model])",len(subset_dataset_two[split][model]))
                subset_dataset_two.targets = -1*subset_dataset_two.targets + self.num_categories + datasetAndInterfaceTwo.num_categories -1
                #print(subset_dataset_two[split][model].targets)
                print("REAL ONLY model", model, "split", split, "len(self.dataset[split][model])", len(self.dataset[split][model]),"len(subset_dataset_two)",len(subset_dataset_two) )
                self.dataset[split][model] = subset_dataset_two + self.dataset[split][model]
                print("REAL ONLY model", model, "split", split, "len(self.dataset[split][model])", len(self.dataset[split][model]))
                #print("real combined database targets", self.dataset[split][model].targets)
                #self.dataset[split][model] = torch.utils.data.ConcatDataset([self.dataset[split][model], datasetAndInterfaceTwo.dataset[split][model]])
                #print(self.dataset[split][model])


        self.categories_list.extend(sub_set_list_cats_two)
        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        print("num cats", self.num_categories)
        self.training_set_size = len(self.dataset['train']['VAE'])
        #print(self.training_set_size)
        self.val_set_size = len(self.dataset['val']['VAE'])
        #print(self.val_set_size)

        if is_plot_true:
            fig1 = plt.figure(figsize=(10, 10))
            x = 0
            r = 5
            c = 5

            for i in range(x,r*c):
                img, cat = self.dataset['train']['VAE'][i]

                img = img.view(28, 28).data
                img = img.numpy()
                ax = fig1.add_subplot(r, c, i-x + 1)
                ax.axis('off')
                ax.set_title(cat)
                ax.imshow(img, cmap='gray_r')

            plt.ioff()
            plt.show()

            fig2 = plt.figure(figsize=(10, 10))
            x = 0
            r = 5
            c = 5

            for i in range(x,r*c):
                img, cat = self.dataset['val']['VAE'][i]

                img = img.view(28, 28).data
                img = img.numpy()
                ax = fig2.add_subplot(r, c, i-x + 1)
                ax.axis('off')
                ax.set_title(cat)
                ax.imshow(img, cmap='gray_r')

            plt.ioff()
            plt.show()

        if is_count_freq:
            subset_dataset = self.dataset['train']['VAE']
            freq_check = {i: 0 for i in range(0, 26)}
            print("length subset_after combining", len(subset_dataset))
            print("checking real datasets...train and VAE")
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1
            print(freq_check)

        if is_count_freq:
            subset_dataset= self.dataset['val']['VAE']
            freq_check = {i: 0 for i in range(0, 26)}
            print("checking real datasets...val and VAE")
            print("length subset_after combining", len(subset_dataset))
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1
            print(freq_check)
