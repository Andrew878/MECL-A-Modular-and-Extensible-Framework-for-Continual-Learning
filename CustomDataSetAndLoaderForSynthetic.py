import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np

class SyntheticDS(Dataset):
    """
    A customized data loader.
    """

    def __init__(self, synthetic_data_list_unique_label, transforms, real_data_to_blend, combined_categories,number_synthetic_categories, original_cat_index_to_new_cat_index_dict):
        """ Intialize the dataset
        """
        self.synthetic_data_list_unique_label = synthetic_data_list_unique_label
        self.real_data_to_blend = real_data_to_blend
        self.number_synthetic_categories = number_synthetic_categories

        self.index_key_fake = [(i,'fake') for i in range(0,len(self.synthetic_data_list_unique_label))]
        self.index_key_real = [(i,'real') for i in range(0,self.real_data_to_blend.__len__())]

        self.index_key = self.index_key_fake+self.index_key_real
        #self.index_to_category = {  self.index_key}

        self.len = len(self.index_key)
        self.transforms = transforms
        self.classes = combined_categories
        self.is_make_dim_adjustment_for_resnet = False
        self.original_cat_index_to_new_cat_index_dict = original_cat_index_to_new_cat_index_dict

    # You must override __getitem__ and __len__
    def __getitem__(self, new_index):
        """ Get a sample from either the synthetic dataset, or the real data dataset
        """
        original_index, real_or_fake = self.index_key[new_index]


        #print("original_index",original_index, "real_or_fake ",real_or_fake)


        if(real_or_fake == 'fake'):
            #print("here 1")
            image, category = self.synthetic_data_list_unique_label[original_index]


            # NEED TO FIX

            image = image.to(dtype=torch.float).cpu()
            # this adjustment is because resnet requires particular dimensions as input
            if not self.is_make_dim_adjustment_for_resnet:
                category = [new_index]
                category = torch.tensor(np.array([category, ])).to(dtype=torch.long).cpu()
            else:
                #print("reached")
                image_pil = transforms.ToPILImage()(image)
                image = self.transforms(image_pil)




        elif (real_or_fake == 'real'):
            #print("here 2")
            image, old_category = self.real_data_to_blend.__getitem__(original_index)
            #category = self.real_data_to_blend.targets[original_index]
            #print(image, category)
            # so that each category has its own label, need to add length of synthetic categories

            category = self.original_cat_index_to_new_cat_index_dict[old_category]


            # for VAE we only want sample one of one (this aligns dimensions from c to [28,28]
            # for CNN we want to maintain the three channels, i.e. so desired inputs are [3,224,224] (224 because of resizing transform)
            #print(image.size(1), image.size())

            if not self.is_make_dim_adjustment_for_resnet:
                image = image[0].float()
                category = [category]
                category = torch.tensor(np.array([category, ])).to(dtype=torch.long)#, device='cuda')



        #print(real_or_fake, image.size(), category)#, image.type(), category,category.size(),  category.type(), category)

        return image, category


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

