import torch
from torch.utils.data import Dataset

import numpy as np


class SyntheticDS(Dataset):
    """
    A customized data loader. This is used when original data and synthetic data are blended into one data set.
    """

    def __init__(self, synthetic_data_list_unique_label, transforms, real_data_to_blend, combined_categories,number_synthetic_categories, original_cat_index_to_new_cat_index_dict):
        """ Intialize the dataset
        """
        self.synthetic_data_list_unique_label = synthetic_data_list_unique_label
        self.real_data_to_blend = real_data_to_blend
        self.number_synthetic_categories = number_synthetic_categories

        # creates an index that maps the new datasets indices to their original source
        self.index_key_fake = [(i,'fake') for i in range(0,len(self.synthetic_data_list_unique_label))]
        self.index_key_real = [(i,'real') for i in range(0,self.real_data_to_blend.__len__())]
        self.index_key = self.index_key_fake+self.index_key_real

        self.len = len(self.index_key)
        self.transforms = transforms
        self.classes = combined_categories

        # this adjustment is because CNN and VAE model inputs are different, so dimension adjustments need to be made
        self.is_make_dim_adjustment_for_resnet = False

        # maps the original category y values to new new label y values
        self.original_cat_index_to_new_cat_index_dict = original_cat_index_to_new_cat_index_dict


    def __getitem__(self, new_index):
        """ Get a sample from either the synthetic dataset, or the real data dataset
        """
        original_index, real_or_fake = self.index_key[new_index]


        # real and fake images are in slightly different forms
        if(real_or_fake == 'fake'):
            image, category = self.synthetic_data_list_unique_label[original_index]

            image = image.to(dtype=torch.float).cpu()

            # this adjustment is because resnet requires particular dimensions as input
            if not self.is_make_dim_adjustment_for_resnet:
                category = [category]
                category = torch.tensor(np.array([category, ])).to(dtype=torch.long).cpu()
            else:
                image = self.transforms['CNN']['test_to_image'](image)

        elif (real_or_fake == 'real'):
            image, old_category = self.real_data_to_blend.__getitem__(original_index)

            # so that each category has its own label, need to add length of synthetic categories

            category = self.original_cat_index_to_new_cat_index_dict[old_category]


            # for VAE we only want sample one of one (this aligns dimensions from c to [28,28]
            # for CNN we want to maintain the three channels, i.e. so desired inputs are [3,224,224] (224 because of resizing transform)
            if not self.is_make_dim_adjustment_for_resnet:
                image = image[0].float()
                category = [category]
                category = torch.tensor(np.array([category, ])).to(dtype=torch.long)#, device='cuda')
            else:

                x_noisy = image
                image = self.transforms['CNN']['test_to_image'](torch.squeeze(x_noisy).detach().numpy())



        return image.float(), category


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

