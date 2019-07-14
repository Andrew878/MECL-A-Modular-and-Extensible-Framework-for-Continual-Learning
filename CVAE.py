from __future__ import print_function, division
import torch
#import torch.nn.sigmoid as sigmoid
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Utils


class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.pre_latent = 200
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(num_channels, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc2 = nn.Linear(1024, self.pre_latent)
        self.fc3 = nn.Linear(n_categories, 1000)
        self.mu = nn.Linear(self.pre_latent, self.latent_dim)
        self.var = nn.Linear(self.pre_latent, self.latent_dim)

    def forward(self, x, labels):
        batch_size = x.size(0)
        #print(x.size())
        x = x.view(batch_size, 1, self.input_dim, self.input_dim)
        #print(x.size())
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64 * self.input_dim * self.input_dim)
        y_ = self.fc3(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # latent parameters
        mean = self.mu(x)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(x)
        # log_var is of shape [batch_size, latent_dim]
        return mean, log_var

    def update_y_layer(self, new_layer):
        self.fc3 = new_layer
        
        



class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):
        self.z_dim = latent_dim
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.fc2 = nn.Linear(n_categories, 1000)
        self.fc = nn.Linear(self.z_dim + 1000, 64 * self.input_dim * self.input_dim)
        # self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, num_channels, 5, 1, 2)
        self.final_layer = nn.Sigmoid()


    def forward(self, x, labels):
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
        self.fc2 = new_layer


class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_categories: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.latent_dim = latent_dim
        self.device = device
        self.n_categories = n_categories
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_categories,num_channels, device).to(device)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, n_categories,num_channels, device).to(device)

    def forward(self, x, y):
        # x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x, y)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(x_sample, y)
        # generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

    def update_y_layers(self, encoder, decoder):
        self.encoder.update_y_layer(encoder)
        self.decoder.update_y_layer(decoder)

    def encode_then_decode_without_randomness(self, x, y):

        # encode
        z_mu, z_var = self.encoder(x,y)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)*0
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(x_sample, y)
        # generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

    def loss(self,x, reconstructed_x, mean, log_var):
        # reconstruction loss
        RCL = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
        # kl divergence loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        #print(RCL)
        #print(KLD,"\n")
        return RCL + KLD

    def get_sample_reconstruction_error_from_single_category_without_randomness(self, x, y, is_random = False, is_already_single_tensor = True):

        if not is_already_single_tensor:
            y = torch.tensor(np.array([[y], ])).to(dtype=torch.long)


        x = x.to(self.device)
        y = Utils.idx2onehot(y, self.n_categories).to(self.device, dtype=x.dtype)

        if is_random:
            reconstructed_x, z_mu, z_var = self.forward(x,y)
        else:
            reconstructed_x, z_mu, z_var = self.encode_then_decode_without_randomness(x,y.float())

        loss = self.loss(x, reconstructed_x, z_mu, z_var)
        return loss

    def get_sample_reconstruction_error_from_all_category(self, x, by_category_mean_std_of_reconstruction_error=None, is_random = False, only_return_best = True, is_standardised_distance_check = False):
        class_with_best_fit = 0
        lowest_error_of_cat = 10000000000
        class_with_best_fit_std_dev = 0
        lowest_error_of_cat_std_dev = 10000000000
        list_by_cat = []

        for category in range(0,self.n_categories):

            loss= self.get_sample_reconstruction_error_from_single_category_without_randomness(x, category,is_random=False,is_already_single_tensor=False)


            info_for_cat = (category, loss)

            if (loss < lowest_error_of_cat):
                lowest_error_of_cat = loss.item()
                class_with_best_fit = category

            if is_standardised_distance_check:
                mean_cat = by_category_mean_std_of_reconstruction_error[category][0]
                std_cat = by_category_mean_std_of_reconstruction_error[category][1]
                std_error_distance = (loss.item() - mean_cat) / std_cat

                if (std_error_distance < lowest_error_of_cat_std_dev):
                    lowest_error_of_cat_std_dev = std_error_distance
                    class_with_best_fit_std_dev = category

                info_for_cat = (info_for_cat, (category,std_error_distance))

            list_by_cat.append(info_for_cat)


        if only_return_best:

            info = (class_with_best_fit, lowest_error_of_cat)
            if is_standardised_distance_check:
                return [info,(class_with_best_fit_std_dev,lowest_error_of_cat_std_dev)]


            return [info]

        else:
            list_by_cat

    def generate_single_random_sample(self, category, is_random_cat = False):

        z = torch.randn(1, self.latent_dim).to(self.device)

        if is_random_cat:
            # pick randomly 1 class, for which we want to generate the data
            y_1d = torch.randint(0, self.n_categories, (1, 1)).to(dtype=torch.long)
        else:
            y_1d = torch.tensor(np.array([[category], ])).to(dtype=torch.long)

        y = Utils.idx2onehot(y_1d, self.n_categories).to(self.device, dtype=z.dtype)

        # don't need gradients
        with torch.no_grad():
            reconstructed_img = self.decoder(z, y)

        img = reconstructed_img.view(28, 28).data
        return img

    def generate_synthetic_set_all_cats(self, number_per_category=1):
        synthetic_data_list_x = []
        synthetic_data_list_y = []

        for n in range(0, number_per_category):
            for category in range(0, self.n_categories):
                img = self.generate_single_random_sample(category,is_random_cat=False)
                synthetic_data_list_x.append(img)
                synthetic_data_list_y.append(category)


        return synthetic_data_list_x, synthetic_data_list_y,


