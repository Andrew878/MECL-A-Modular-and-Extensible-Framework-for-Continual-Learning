from __future__ import print_function, division
import torch
#import torch.nn.sigmoid as sigmoid
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Utils


"""Contains all classes and methods related to the conditional VAE (i.e. encoder, decoder)"""

class Encoder(nn.Module):
    """VAE encoder. """

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):
        """"
            input_dim: A integer for input dimension size.
            latent_dim: A integer for latent dimension size.
            n_categories: A integer for number of classes.
            num_channels: channel size of input (e.g. grayscale == 1)
        """
        super().__init__()

        self.pre_latent = 200
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)

        # fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc2 = nn.Linear(1024, self.pre_latent)
        self.fc3 = nn.Linear(n_categories, 1000)

        # parameters of Gaussian approximation for latent distributions
        self.mu = nn.Linear(self.pre_latent, self.latent_dim)
        self.var = nn.Linear(self.pre_latent, self.latent_dim)

    def forward(self, x, labels):
        """Forward pass of encoder"""
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.input_dim, self.input_dim)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64 * self.input_dim * self.input_dim)
        y_ = self.fc3(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        mean = self.mu(x)
        log_var = self.var(x)
        return mean, log_var

    def update_y_layer(self, new_layer):
        """Used for when layers are replaced (e.g. adding a new category or applying transfer learning)"""
        self.fc3 = new_layer
        
        



class Decoder(nn.Module):
    """VAE encoder. """

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):
        """"
                    input_dim: A integer for input dimension size.
                    latent_dim: A integer for latent dimension size.
                    n_categories: A integer for number of classes.
                    num_channels: channel size of input (e.g. grayscale == 1)
        """
        self.z_dim = latent_dim
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.fc2 = nn.Linear(n_categories, 1000)
        self.fc = nn.Linear(self.z_dim + 1000, 64 * self.input_dim * self.input_dim)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, num_channels, 5, 1, 2)
        self.final_layer = nn.Sigmoid()


    def forward(self, x, labels):
        """Forward pass of decoder"""
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, self.input_dim, self.input_dim)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = self.final_layer(x)
        return x

    def update_y_layer(self, new_layer):
        """Used for when layers are replaced (e.g. adding a new category or applying transfer learning)"""
        self.fc2 = new_layer


class CVAE(nn.Module):
    """ This the VAE, which creates a encoder and decoder.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):

        """"
                    input_dim: A integer for input dimension size.
                    latent_dim: A integer for latent dimension size.
                    n_categories: A integer for number of classes.
                    num_channels: channel size of input (e.g. grayscale == 1)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.device = device
        self.n_categories = n_categories

        # create encoder, create decoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_categories,num_channels, device).to(device)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, n_categories,num_channels, device).to(device)

    def forward(self, x, y):
        """Forward pass of VAE"""

        # first forward pass of encoder
        z_mu, z_var = self.encoder(x, y)

        # sample from the distribution having latent parameters z_mu, z_var
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # now forward pass of decoder
        generated_x = self.decoder(x_sample, y)
        return generated_x, z_mu, z_var

    def update_y_layers(self, encoder, decoder):
        """Update y input layers (e.g. for adding a new label)"""
        self.encoder.update_y_layer(encoder)
        self.decoder.update_y_layer(decoder)

    def encode_then_decode_without_randomness(self, x, y):
        """A forward pass but without a random sampling process"""

        # encode
        z_mu, z_var = self.encoder(x,y)

        # sample from the distribution having latent parameters z_mu, z_var
        # note, only taking the mean here.
        std = torch.exp(z_var / 2)*0
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        generated_x = self.decoder(x_sample, y)
        return generated_x, z_mu, z_var

    def loss(self,x, reconstructed_x, mean, log_var):
        """Calculate the loss. Loss = Reconstruction error plus KL divergence"""

        # reconstruction loss (input, target)
        reconstructed_x = torch.squeeze(reconstructed_x,1)
        x = torch.squeeze(x,1)
        RCL = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')

        # kl divergence loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return RCL + KLD

    def get_sample_reconstruction_error_from_single_category_without_randomness(self, x, y, is_random = False, is_already_single_tensor = True):

        """Obtains reconstruction error from a sample with a specific category and without randomness"""
        if not is_already_single_tensor:
            y = torch.tensor(np.array([[y], ])).to(dtype=torch.long)

        # send to device and create one-hot y vector
        x = x.to(self.device)
        y = Utils.idx2onehot(y, self.n_categories).to(self.device, dtype=x.dtype)

        # forward passes with and without randomness
        if is_random:
            reconstructed_x, z_mu, z_var = self.forward(x,y)
        else:
            reconstructed_x, z_mu, z_var = self.encode_then_decode_without_randomness(x,y.float())

        # calculate loss and return
        loss = self.loss(x, reconstructed_x, z_mu, z_var)
        return loss, reconstructed_x

    def get_sample_reconstruction_error_from_all_category(self, x, by_category_mean_std_of_reconstruction_error=None, is_random = False, only_return_best = True, is_standardised_distance_check = False):

        """Obtains reconstruction error from a sample from ALL categories, and choices the best. Can be done with and without randomness"""

        # intialise variables
        class_with_best_fit = 0
        lowest_error_of_cat = 10000000000
        class_with_best_fit_std_dev = 0
        lowest_error_of_cat_std_dev = 10000000000
        list_by_cat = []

        # send models to GPU
        self.send_all_to_GPU()


        # cycle through the different categories, and choose the one with lowest reconstruction error
        for category in range(0,self.n_categories):

            # get reconstruction error for category
            loss, reconstructed_x = self.get_sample_reconstruction_error_from_single_category_without_randomness(x, category,is_random=False,is_already_single_tensor=False)

            # place info in data structure
            info_for_cat = (category, loss)

            # check for best with absolute measure
            if (loss < lowest_error_of_cat):
                lowest_error_of_cat = loss.item()
                class_with_best_fit = category

            # check for best with relative measure using the previously recorded mean and std recon error for training set.
            if is_standardised_distance_check:
                mean_cat = by_category_mean_std_of_reconstruction_error[category][0]
                std_cat = by_category_mean_std_of_reconstruction_error[category][1]
                std_error_distance = (loss.item() - mean_cat) / std_cat

                if (std_error_distance < lowest_error_of_cat_std_dev):
                    lowest_error_of_cat_std_dev = std_error_distance
                    class_with_best_fit_std_dev = category

                info_for_cat = (info_for_cat, (category,std_error_distance))

            list_by_cat.append(info_for_cat)

        # send models back to CPU
        self.send_all_to_CPU()

        # user can specify if all reconstruction error info is returned, or only the best (lowest).
        # Different information is also returned depending on distance measure used
        if only_return_best:

            info = (class_with_best_fit, lowest_error_of_cat)
            if is_standardised_distance_check:
                return [info,(class_with_best_fit_std_dev,lowest_error_of_cat_std_dev)],reconstructed_x

            return [info], reconstructed_x

        else:
            return list_by_cat, reconstructed_x



    def generate_single_random_sample(self, category, z=None,is_random_cat = False):

        """Generates a single synthetic sample. Z is either provided or generated within this method"""

        # if Z not provided, generate a random z
        if z is None:
            z = torch.randn(1, self.latent_dim).to(self.device)

        # pick randomly one class, for which we want to generate the data.
        if is_random_cat:
            y_1d = torch.randint(0, self.n_categories, (1, 1)).to(dtype=torch.long)
        # Otherwise use provided class
        else:
            y_1d = torch.tensor(np.array([[category], ])).to(dtype=torch.long)

        # convert to one-hot encoding
        y = Utils.idx2onehot(y_1d, self.n_categories).to(self.device, dtype=z.dtype)

        # use decoder to get sample. we don't need gradients
        with torch.no_grad():
            reconstructed_img = self.decoder(z, y)

        # reshape and return
        img = reconstructed_img.view(28, 28).data
        return img

    def generate_synthetic_set_all_cats(self, synthetic_data_list_unique_label = None, number_per_category=1, is_store_on_CPU = False):

        """Generate random samples for all categories. Used for Pseudo-rehearsal"""
        synthetic_data_list_x = []
        synthetic_data_list_y = []

        self.send_all_to_GPU()

        # cycle through each category and obtain the requested number of synthetic samples
        for n in range(0, number_per_category):
            for category in range(0, self.n_categories):
                img = self.generate_single_random_sample(category,is_random_cat=False)

                # keeps samples on CPU to prevent exceeding GPU memory
                if(is_store_on_CPU):
                    img = img.cpu()
                synthetic_data_list_x.append(img)
                synthetic_data_list_y.append(category)

                # give the sample a unique ID for later identification
                if synthetic_data_list_unique_label != None:
                    synthetic_data_list_unique_label.append((img, category))

        self.send_all_to_CPU()

        return synthetic_data_list_x, synthetic_data_list_y

    def send_all_to_GPU(self):
        """Send models to GPU"""
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def send_all_to_CPU(self):
        """Send models to CPU"""
        self.encoder.cpu()
        self.decoder.cpu()