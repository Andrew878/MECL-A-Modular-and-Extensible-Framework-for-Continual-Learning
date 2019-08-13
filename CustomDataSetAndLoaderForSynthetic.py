import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import Utils
from scipy.ndimage import gaussian_filter

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


        #self.freq_check = {i:0 for i in range(0,30)}
        #self.freq = 0

    # You must override __getitem__ and __len__
    def __getitem__(self, new_index):
        """ Get a sample from either the synthetic dataset, or the real data dataset
        """
        original_index, real_or_fake = self.index_key[new_index]

        self.freq += 1
        #print("original_index",original_index, "real_or_fake ",real_or_fake)


        # real and fake images are in slightly different forms
        if(real_or_fake == 'fake'):
            #print("here 1")
            image, category = self.synthetic_data_list_unique_label[original_index]

            #self.freq_check[category] += 1
            # NEED TO FIX

            image = image.to(dtype=torch.float).cpu()
            # this adjustment is because resnet requires particular dimensions as input
            if not self.is_make_dim_adjustment_for_resnet:
                #if(category==10):
                    #print("category 10 found")
                category = [category]
                category = torch.tensor(np.array([category, ])).to(dtype=torch.long).cpu()
            else:
                #print("reached")
                image = self.transforms['CNN']['test_to_image'](image)

        elif (real_or_fake == 'real'):
            #print("here 2")
            image, old_category = self.real_data_to_blend.__getitem__(original_index)
            #category = self.real_data_to_blend.targets[original_index]
            #print(image, category)
            # so that each category has its own label, need to add length of synthetic categories

            category = self.original_cat_index_to_new_cat_index_dict[old_category]

            #self.freq_check[category] += 1

            # for VAE we only want sample one of one (this aligns dimensions from c to [28,28]
            # for CNN we want to maintain the three channels, i.e. so desired inputs are [3,224,224] (224 because of resizing transform)
            #print(image.size(1), image.size())


            if not self.is_make_dim_adjustment_for_resnet:
                image = image[0].float()
                category = [category]
                category = torch.tensor(np.array([category, ])).to(dtype=torch.long)#, device='cuda')
            else:

                x_noisy = image

                #x_noisy = gaussian_filter(image.cpu().detach().numpy(), sigma=.5)
                #x_noisy = torch.Tensor(x_noisy).cpu()

                image = self.transforms['CNN']['test_to_image'](torch.squeeze(x_noisy).detach().numpy())
                #image = torch.unsqueeze(x, 0)
                # category = [category]
                # category = torch.tensor(np.array([category, ])).to(dtype=torch.long)



        #if self.freq%1000000==0:
        #    print(self.freq, self.freq_check)
        #print(real_or_fake, image.size(), category, image.type(), category)#,category.size(),  category.type(), category)

        return image.float(), category


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

