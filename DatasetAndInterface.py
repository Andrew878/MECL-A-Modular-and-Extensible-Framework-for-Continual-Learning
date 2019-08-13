
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import copy
import matplotlib.pyplot as plt
from Invert import Invert


class DataSetAndInterface:

    """
    A class that holds datasets, their transformations, and other helpful variables (such as index to category dictionaries).
    Also contains methods that merge datasets and provide subsets of datasets.
    """

    def __init__(self, name, dataset, path, transformations, original_channel_number,original_input_dimensions, list_of_fixed_noise):

        self.name = name
        self.dataset = dataset
        # original dataset is copied for use in testing/evaluation
        self.dataset_original = copy.deepcopy(dataset)
        self.path = path
        # contant fixed noise that to help visualise changes in synthetic sample quality
        self.list_of_fixed_noise = list_of_fixed_noise

        self.transformations = transformations
        self.dataset_splits = ['train', 'val']

        # Obtaining class labels
        # The EMNIST class labels in the torchvision module are incorrect. This hack fixes it.
        if(name == 'EMNIST'):
            self.categories_list = [chr(i) for i in range(ord('a'), ord('z')+1)]

        elif (name == 'SVHN'):
            self.categories_list = [str(i)+" SVHN" for i in range(0, 10)]
        else:
            self.categories_list = dataset['train']['VAE'].classes

        # plot examples of dataset
        self.show_plots_of_dataset()

        # match index to category
        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        self.original_channel_number = original_channel_number
        self.original_input_dimensions = original_input_dimensions
        self.training_set_size = len(dataset['train']['VAE'])
        print(name, " training set size is ",self.training_set_size)
        self.val_set_size = len(dataset['val']['VAE'])
        print(name, " test set size is ",self.val_set_size)

        # the number of times the dataset has been changed
        self.mutated_count = 0

    def show_plots_of_dataset(self):
        """Plot 100 samples of training dataset."""

        fig1 = plt.figure(figsize=(15, 15))
        x = 0
        r = 10
        c = 10
        for i in range(x, r * c):
            img, cat = self.dataset['train']['VAE'][i]
            img = img.view(28, 28).data
            img = img.numpy()
            ax = fig1.add_subplot(r, c, i - x + 1)
            ax.axis('off')
            ax.imshow(img,cmap='gray')
        plt.ioff()
        plt.show()

    def update_data_set(self, dataset, is_synthetic_change = True):

        """After adding synthetic samples, or merging a dataset, important class variables need to be refreshed"""

        if is_synthetic_change:
            self.mutated_count += 1
        self.dataset = dataset
        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        self.training_set_size = len(dataset['train']['VAE'])
        print(self.name, " training set size is ",self.training_set_size)
        self.val_set_size = len(dataset['val']['VAE'])
        print(self.name, " test set size is ",self.val_set_size)

    def reset_variables_to_initial_state(self):

        """Returns the object to its original state. Many of the tests/evaluations require amending datasets by reducing or blending.
        This method is resets the object so another test can be performed."""

        # Most of below is identical to initialisation of function

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

        """Returns a dictionary of dataloaders for training or evaluation. Branch component input refers to CNN or VAE"""

        if (dataset_selected == None):
            dataset_selected = self.dataset

        if (split == None):
            dataset_splits = self.dataset_splits
        else:
            data_loader = torch.utils.data.DataLoader(dataset_selected, batch_size=BATCH_SIZE,shuffle=True, num_workers=num_workers)
            return data_loader

        # cycle through each split (train/test)
        dataloaders = {}
        for d_set in dataset_splits:

            dataloaders[d_set] = torch.utils.data.DataLoader(dataset_selected[d_set][branch_component], batch_size=BATCH_SIZE,shuffle=True, num_workers=num_workers)

            # there is a dimension discrepancy between what resnet requires and what the VAE outputs.
            # This fixes it in the CustomDataSetAndLoader class
            if self.mutated_count > 0 and branch_component == 'CNN':
                dataset_selected[d_set][branch_component].is_make_dim_adjustment_for_resnet = True

        return dataloaders

    def obtain_dataset_with_subset_of_categories(self, branch_component, split, category_subset):

        """
        Returns a copy of the original datasets but with subset of categories removed.
        This code was adapted from a post on the Pytorch forums. the https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276
        """

        # flags that call upon helper code
        is_count_freq = False
        is_plot_output = False

        category_subset_indices = [self.label_to_index_dict[i] for i in category_subset]
        subset_dataset = copy.deepcopy(self.dataset[split][branch_component])

        print("********** Preparing subset dataset",branch_component, split)
        print("length subset before ", len(subset_dataset))

        # counts the frequency of classes before reduction
        if is_count_freq:
            freq_check = {i: 0 for i in range(0, 26)}
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1


        # cycle through for each category desired, and make a record of their index location
        idx = torch.zeros(len(subset_dataset),dtype=torch.uint8)
        for cat_index in category_subset_indices:

            # get indices for this category
            if self.name == 'SVHN':
                id_cat = torch.tensor(subset_dataset.labels) == cat_index
            else:
                id_cat = subset_dataset.targets == cat_index

            # accumulate these indices across categories
            idx += id_cat

        # the SVHN dataset has a slightly different format to other datasets. This code accounts for it
        if self.name == 'SVHN':
            subset_dataset.labels = torch.tensor(subset_dataset.labels)[idx]
            subset_dataset.labels = subset_dataset.labels.numpy()
            subset_dataset.data = torch.tensor(subset_dataset.data)[idx]
            subset_dataset.data = subset_dataset.data.numpy()
        else:
            subset_dataset.targets = subset_dataset.targets[idx]
            subset_dataset.data = subset_dataset.data[idx]



        # the code below is just for debugging purposes
        # counts the frequency of classes after reduction
        if is_count_freq:
            freq_check = {i: 0 for i in range(0, 26)}
            print("length subset after", len(subset_dataset))
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1
            print(freq_check)

        # plots a sample of images
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


        return subset_dataset


    def obtain_dataloader_with_subset_of_categories(self, branch_component, split, category_subset, BATCH_SIZE = 64, num_workers=4):

        """Returns dataloader with subset of categories"""

        return self.return_data_loaders(branch_component,self.obtain_dataset_with_subset_of_categories(branch_component,split, category_subset),BATCH_SIZE,num_workers,category_subset)


    def add_outside_data_to_data_set(self, datasetAndInterfaceTwo, sub_set_list_cats_two=[]):

        """Merges a dataset (i.e. DataSetTwo) from outside this object. This is used for evaluating the continual learning functionality
        that adds new categories"""

        # flags that call upon helper code
        is_count_freq = True
        is_plot_true = True

        self.training_set_size = len(self.dataset['train']['VAE'])
        self.val_set_size = len(self.dataset['val']['VAE'])

        # counting frequencies before acdding datasets
        if is_count_freq:
            subset_dataset = self.dataset['train']['VAE']
            freq_check = {i: 0 for i in range(0, 26)}
            print("length subset before", len(subset_dataset))
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1
            print(freq_check)


        # if specific labels aren't given, then add all
        if len(sub_set_list_cats_two) == 0:
            sub_set_list_cats_two = datasetAndInterfaceTwo.categories_list

        # cycle through dataset for each model type
        for model in ['VAE', 'CNN']:

            # cycle through split for each model type
            for split in self.dataset_splits:

                # obtain a reduced category dataset
                subset_dataset_two = datasetAndInterfaceTwo.obtain_dataset_with_subset_of_categories(model, split, sub_set_list_cats_two)

                # amend the labels of the previous dataset (this is because in the combined dataset, they will no longer begin at 0)
                subset_dataset_two.targets = -1*subset_dataset_two.targets + self.num_categories + datasetAndInterfaceTwo.num_categories -1

                # add old and new datasets together
                self.dataset[split][model] = subset_dataset_two + self.dataset[split][model]

        # revise some important class variables
        self.categories_list.extend(sub_set_list_cats_two)
        self.label_to_index_dict = {k: v for v, k in enumerate(self.categories_list)}
        self.num_categories = len(self.categories_list)
        self.training_set_size = len(self.dataset['train']['VAE'])
        self.val_set_size = len(self.dataset['val']['VAE'])


        # the below only relates to helper code
        # plots a sample of images
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

        # counting frequencies after adding datasets
        if is_count_freq:
            subset_dataset = self.dataset['train']['VAE']
            freq_check = {i: 0 for i in range(0, 26)}
            print("length subset_after combining", len(subset_dataset))
            print("checking real datasets...train and VAE")
            for i in range(0, len(subset_dataset)):
                image, cat = subset_dataset[i]
                freq_check[cat] += 1
            print(freq_check)



    def reduce_categories_in_dataset(self,sub_set_list=[]):

        """Reduces dataset to a smaller subset of categories. Required for several tests/evaluation procedures.
        Combines several steps from other methods into a single, clean method"""

        print("Reducing from",self.categories_list,"to",sub_set_list)

        self.categories_list = sub_set_list

        # cycle through each model and dataset split and reduce categories
        for model in ['VAE', 'CNN']:
            for split in self.dataset_splits:
                subset_dataset_two = self.obtain_dataset_with_subset_of_categories(model, split, sub_set_list)
                self.dataset[split][model] = subset_dataset_two

        self.update_data_set(self.dataset,False)



    def update_transformations(self, new_transformation,  is_all = True):

        """Required to continual learning evaluation. Inserts a new transformation to replicate concept drift.
        Due to the source data, SVHN and EMNIST require their own unique transformations.
        """

        normalise_for_PIL_mean = (0.5, 0.5, 0.5)
        normalise_for_PIL_std = (0.5, 0.5, 0.5)
        image_height_MNIST = 28

        if (self.name == 'EMNIST'):
            self.transformations['CNN'] = {
                'train': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    new_transformation,
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'val': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    new_transformation,
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    new_transformation,
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test_to_image': transforms.Compose([
                    transforms.ToPILImage(),
                    new_transformation,
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
            }

            self.transformations['VAE']= {
                'train': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    new_transformation,
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    new_transformation,
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
                'test': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    new_transformation,
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
            }


        elif (self.name == 'SVHN'):

            self.transformations['CNN'] = {
                'train': transforms.Compose([
                    transforms.Grayscale(),
                    new_transformation,
                    Invert(),
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'val': transforms.Compose([
                    transforms.Grayscale(),
                    new_transformation,
                    Invert(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test': transforms.Compose([
                    transforms.Grayscale(),
                    new_transformation,
                    Invert(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test_to_image': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(),
                    new_transformation,
                    Invert(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
            }
            self.transformations['VAE'] = {
                'train': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    new_transformation,
                    Invert(),
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    new_transformation,
                    Invert(),
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
                'test': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    new_transformation,
                    Invert(),

                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
            }


        else:
            self.transformations['CNN'] = {
                'train': transforms.Compose([
                    new_transformation,
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'val': transforms.Compose([
                    new_transformation,
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test': transforms.Compose([
                    new_transformation,
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test_to_image': transforms.Compose([
                    transforms.ToPILImage(),
                    new_transformation,
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
            }

            self.transformations['VAE'] = {
                'train': transforms.Compose([
                    new_transformation,
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    new_transformation,
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
                'test': transforms.Compose([
                    new_transformation,
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
            }


        for key1 in self.dataset:
            for key2 in self.dataset[key1]:
                self.dataset[key1][key2].transform = self.transformations[key2][key1]
