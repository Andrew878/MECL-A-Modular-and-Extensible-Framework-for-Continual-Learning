import torch
import torch.optim
import CVAE
import Utils
import time
import copy
from torchvision import models
import torch.nn as nn
import numpy as np
import mpu


class TaskBranch:

    def __init__(self, task_name, dataset_interface, device, initial_directory_path):

        self.task_name = task_name
        self.dataset_interface = dataset_interface
        self.initial_directory_path = initial_directory_path
        self.num_categories_in_task = self.dataset_interface.num_categories
        self.categories_list = self.dataset_interface.categories_list

        self.VAE_most_recent = None
        self.VAE_history = []
        self.CNN_most_recent = None
        self.CNN_history = []

        self.device = device
        self.VAE_optimizer = None
        self.CNN_optimizer = None

        self.synthetic_samples_for_reuse = {'train':[], 'val':[]}
        self.by_category_record_of_reconstruction_error = None
        self.by_category_mean_std_of_reconstruction_error = None
        self.has_VAE_changed_since_last_mean_std_measurements = True

    def create_and_train_VAE(self, model_id, num_epochs=30, batch_size=64, hidden_dim=10, latent_dim=75,
                             is_synthetic=False, is_take_existing_VAE=False, teacher_VAE=None, new_categories_to_add_to_existing_task = [], is_completely_new_task = False,
                             epoch_improvement_limit=20, learning_rate=0.00035, betas=(0.5, .999), is_save=False,):

        self.synthetic_samples_for_reuse = []

        # if not using a pre-trained VAE, create new VAE
        if not is_take_existing_VAE:
            # need to fix hidden dimensions
            self.VAE_most_recent = CVAE.CVAE(self.dataset_interface.original_input_dimensions, hidden_dim, latent_dim,
                                             self.num_categories_in_task,
                                             self.dataset_interface.original_channel_number, self.device)
        # if using a pre-trained VAE
        else:
            self.VAE_most_recent = copy.deepcopy(teacher_VAE).to(self.device)

            # if adding a new task category to learn
            if (is_completely_new_task or len(new_categories_to_add_to_existing_task) != 0):
                fc3 = nn.Linear(self.num_categories_in_task + len(new_categories_to_add_to_existing_task), 1000)
                fc2 = nn.Linear(self.num_categories_in_task + len(new_categories_to_add_to_existing_task), 1000)
                self.VAE_most_recent.update_y_layers(fc3, fc2)
                self.VAE_most_recent.to(self.device)

        patience_counter = 0
        best_val_loss = 100000000000
        self.VAE_optimizer = torch.optim.Adam(self.VAE_most_recent.parameters(), lr=learning_rate, betas=betas)

        if (len(new_categories_to_add_to_existing_task) == 0):
            dataloaders = self.dataset_interface.return_data_loaders('VAE', BATCH_SIZE=batch_size)
        else:
            dataloaders = self.dataset_interface.obtain_dataloader_with_subset_of_categories('VAE', new_categories_to_add_to_existing_task, BATCH_SIZE=batch_size)


        start_timer = time.time()
        for epoch in range(num_epochs):

            train_loss = self.run_a_VAE_epoch_calculate_loss(dataloaders['train'], is_train=True,
                                                             is_synthetic=is_synthetic)
            val_loss = self.run_a_VAE_epoch_calculate_loss(dataloaders['val'], is_train=False,
                                                           is_synthetic=is_synthetic)

            train_loss /= self.dataset_interface.training_set_size
            val_loss /= self.dataset_interface.val_set_size
            time_elapsed = time.time() - start_timer

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                patience_counter = 1

                # send to CPU to save on GPU RAM
                best_model_wts = copy.deepcopy(self.VAE_most_recent.state_dict())
                print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f} ', time_elapsed,
                      "**** new best ****")


            else:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f} ', time_elapsed)
                patience_counter += 1

            if patience_counter > epoch_improvement_limit:
                break

        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_val_loss))

        self.VAE_most_recent.load_state_dict(best_model_wts)

        model_string = "VAE "+ self.task_name +" epochs" + str(num_epochs) + ",batch" + str(batch_size) + ",z_d" + str(
            latent_dim) + ",synth" + str(is_synthetic) + ",rebuilt" + str(is_take_existing_VAE) + ",lr" + str(
            learning_rate) + ",betas" + str(betas) + "lowest_error "+str(best_val_loss)+" "
        model_string += model_id


        if is_save:
            torch.save(self.VAE_most_recent, self.initial_directory_path + model_string)

        self.obtain_VAE_recon_error_mean_and_std_per_class(self.initial_directory_path + model_string)

        self.VAE_most_recent.cpu()
        torch.cuda.empty_cache()

    def run_a_VAE_epoch_calculate_loss(self, data_loader, is_train, teacher_VAE=None, is_synthetic=False,
                                       is_cat_info_required=False):

        # To update record for mean and std deviation distance. This deletes old entries before calculating
        if is_cat_info_required:
            self.by_category_record_of_reconstruction_error = {i: [] for i in range(0, self.num_categories_in_task)}

        loss_sum = 0
        for i, (x, y) in enumerate(data_loader):
            # reshape the data into [batch_size, 784]
            # print(x.size())
            # x = x.view(batch_size, 1, 28, 28)
            x = x.to(self.device)
            # print(x)

            if is_cat_info_required:
                cat_record = self.by_category_record_of_reconstruction_error[y.item()]

            # convert y into one-hot encoding
            y = Utils.idx2onehot(y.view(-1, 1), self.num_categories_in_task)
            y = y.to(self.device)

            # get synthetic samples, and blend them into the batch
            if is_synthetic and teacher_VAE != None:
                synthetic_samples_x, synthetic_samples_y = teacher_VAE.generate_synthetic_set_all_cats(
                    number_per_category=data_loader.batch_size)
                if(is_train):
                    self.synthetic_samples_for_resuse['train'].append((synthetic_samples_x, synthetic_samples_y))
                else:
                    self.synthetic_samples_for_resuse['val'].append((synthetic_samples_x, synthetic_samples_y))

                synthetic_samples_x.append(x)
                x = torch.cat((tuple(synthetic_samples_x)), dim=0).to(self.device)
                synthetic_samples_y.append(y)
                y = torch.cat((tuple(synthetic_samples_y)), dim=0).to(self.device)

            # update the gradients to zero
            if is_train:
                self.VAE_optimizer.zero_grad()

            # forward pass
            # track history if only in train
            with torch.set_grad_enabled(is_train):
                reconstructed_x, z_mu, z_var = self.VAE_most_recent(x, y)

            # loss
            loss = self.VAE_most_recent.loss(x, reconstructed_x, z_mu, z_var)
            loss_sum += loss.item()
            # print(loss.item())

            if is_cat_info_required:
                cat_record.append(loss.item())

            if (is_train):
                loss.backward()
                self.VAE_optimizer.step()

        return loss_sum

    # def generate_synthetic_batch(self, batch_size = 64):

    def load_existing_VAE(self, PATH, is_calculate_mean_std = False):
        self.VAE_most_recent = torch.load(PATH)
        if is_calculate_mean_std:
            print(self.task_name," - Loaded VAE model, now calculating reconstruction error mean, std")
            self.obtain_VAE_recon_error_mean_and_std_per_class(PATH)
        else:
            self.by_category_mean_std_of_reconstruction_error = mpu.io.read(PATH+"mean,std.pickle")

    def load_existing_VAE(self, PATH, is_calculate_mean_std = False):
        self.VAE_most_recent = torch.load(PATH)
        if is_calculate_mean_std:
            print(self.task_name," - Loaded VAE model, now calculating reconstruction error mean, std")
            self.obtain_VAE_recon_error_mean_and_std_per_class(PATH)
        else:
            self.by_category_mean_std_of_reconstruction_error = mpu.io.read(PATH+"mean,std.pickle")


    def create_and_train_CNN(self, model_id, num_epochs=30, batch_size=64, is_frozen=False, is_off_shelf_model=False,
                             epoch_improvement_limit=20, learning_rate=0.0003, betas=(0.999, .999), is_save=False):

        # need to fix hidden dimensions

        if is_off_shelf_model:
            self.CNN_most_recent = models.resnet18(pretrained=True)

            if is_frozen == True:
                for param in self.CNN_most_recent.parameters():
                    param.requires_grad = False

            # take number of features from last layer
            num_ftrs = self.CNN_most_recent.fc.in_features

            # create a new fully connected final layer for training
            self.CNN_most_recent.fc = nn.Linear(num_ftrs, self.num_categories_in_task)

            if is_frozen == True:
                # only fc layer being optimised
                self.CNN_optimizer = torch.optim.Adam(self.CNN_most_recent.fc.parameters(), lr=learning_rate,
                                                      betas=betas)
            else:
                # all layers being optimised
                self.CNN_optimizer = torch.optim.Adam(self.CNN_most_recent.parameters(), lr=learning_rate, betas=betas)

        else:
            self.CNN_most_recent = None
            # all layers being optimised
            self.CNN_optimizer = torch.optim.Adam(self.CNN_most_recent.parameters(), lr=learning_rate, betas=betas)

        criterion = nn.CrossEntropyLoss()

        dataloaders = self.dataset_interface.return_data_loaders('CNN', BATCH_SIZE=batch_size)

        start_timer = time.time()
        best_val_acc = 0.0
        self.CNN_most_recent.to(self.device)
        patience_counter = 0

        for epoch in range(num_epochs):

            train_loss, correct_guesses_train = self.run_a_CNN_epoch_calculate_loss(dataloaders['train'], criterion,
                                                                                    is_train=True)
            val_loss, correct_guesses_val = self.run_a_CNN_epoch_calculate_loss(dataloaders['val'], criterion,
                                                                                is_train=False)

            train_loss /= self.dataset_interface.training_set_size
            val_loss /= self.dataset_interface.val_set_size

            train_acc = correct_guesses_train / self.dataset_interface.training_set_size
            val_acc = correct_guesses_val / self.dataset_interface.val_set_size

            time_elapsed = time.time() - start_timer

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                patience_counter = 1

                # send to CPU to save on GPU RAM
                best_model_wts = copy.deepcopy(self.CNN_most_recent.state_dict())
                print(f'Epoch {epoch}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} ', time_elapsed,
                      "**** new best ****")


            else:
                print(f'Epoch {epoch}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} ', time_elapsed)
                patience_counter += 1

            if patience_counter > epoch_improvement_limit:
                break

        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}'.format(best_val_acc))

        self.CNN_most_recent.load_state_dict(best_model_wts)

        if is_save:
            model_string = "CNN " + self.task_name + " epochs" + str(num_epochs) + ",batch" + str(
                batch_size) + ",pretrained" + str(is_off_shelf_model) + ",frozen" + str(is_frozen) + ",lr" + str(
                learning_rate) + ",betas" + str(betas) +" accuracy "+str(best_val_acc) + " "
            model_string += model_id

            torch.save(self.CNN_most_recent, self.initial_directory_path + model_string)

        self.CNN_most_recent.cpu()
        torch.cuda.empty_cache()

    def obtain_VAE_recon_error_mean_and_std_per_class(self, VAE_MODEL_PATH = None):

        print("Updating VAE record means, standard deviations...")

        data_loader = self.dataset_interface.return_data_loaders(branch_component='VAE',
                                                                 BATCH_SIZE=1)

        # get a record of loss reconstruction on training set, but don't do any training
        self.run_a_VAE_epoch_calculate_loss(data_loader['train'], is_train=False, is_cat_info_required=True)

        # initialize dictionary
        self.by_category_mean_std_of_reconstruction_error = {}

        for cat in range(0, self.num_categories_in_task):
            mean = np.asarray(self.by_category_record_of_reconstruction_error[cat]).mean(axis=0)
            std = np.asarray(self.by_category_record_of_reconstruction_error[cat]).std(axis=0)
            self.by_category_mean_std_of_reconstruction_error[cat] = (mean, std)

        mpu.io.write(VAE_MODEL_PATH+"mean,std.pickle", self.by_category_mean_std_of_reconstruction_error)


    #def

    def given_observation_find_lowest_reconstruction_error(self, x, is_standardised_distance_check=True):

        return self.VAE_most_recent.get_sample_reconstruction_error_from_all_category(x,self.by_category_mean_std_of_reconstruction_error,
                                                                                      is_random=False,
                                                                                      only_return_best=True,
                                                                                      is_standardised_distance_check=is_standardised_distance_check)

    def generate_single_random_sample(self, category, is_random_cat=False):
        return self.VAE_most_recent.generate_single_random_sample(category, is_random_cat)

    def run_a_CNN_epoch_calculate_loss(self, data_loader, criterion, is_train):

        running_loss = 0.0
        running_corrects = 0

        if is_train:
            self.CNN_most_recent.train()
        else:
            self.CNN_most_recent.eval()

        loss_sum = 0.0
        running_corrects = 0
        for i, (x, y) in enumerate(data_loader):
            # reshape the data into [batch_size, 784]
            # print(x.size())
            # x = x.view(batch_size, 1, 28, 28)
            x = x.to(self.device)
            y = y.to(self.device)
            # print(x.size())

            # convert y into one-hot encoding
            # y_one_hot = Utils.idx2onehot(y.view(-1, 1), self.num_categories_in_task)
            # y_one_hot.to(self.device)

            # update the gradients to zero
            self.CNN_optimizer.zero_grad()

            # forward pass
            # track history if only in train
            with torch.set_grad_enabled(is_train):
                outputs = self.CNN_most_recent(x)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs.to(self.device), y)

            # loss
            loss_sum += loss.item()
            running_corrects += torch.sum(preds == y.data)

            # backward + optimize only if in training phase
            if is_train:
                loss.backward()
                self.CNN_optimizer.step()

        return loss_sum, running_corrects.double()

    def load_existing_CNN(self, PATH):
        self.CNN_most_recent = torch.load(PATH)

    def classify_with_CNN(self, x, is_output_one_hot=False):

        output = self.CNN_most_recent(x)
        preds = output

        if not is_output_one_hot:
            max_value, preds = torch.max(output, 1)

        return preds
