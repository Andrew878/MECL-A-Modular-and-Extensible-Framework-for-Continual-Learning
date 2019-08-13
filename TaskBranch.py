import torch
import torch.optim
import torchvision.transforms.functional as TF
import CVAE
import math
import Utils
import time
import copy
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
import mpu
import CustomDataSetAndLoaderForSynthetic
import matplotlib.pyplot as plt
import EpochRecord
import queue

import RecordKeeper
from scipy.ndimage import gaussian_filter


class TaskBranch:

    def __init__(self, task_name, dataset_interface, device, initial_directory_path, record_keeper):

        self.task_name = task_name
        self.dataset_interface = dataset_interface
        self.initial_directory_path = initial_directory_path
        self.num_categories_in_task = self.dataset_interface.num_categories
        self.categories_list = self.dataset_interface.categories_list

        self.VAE_most_recent = None
        self.VAE_most_recent_path = None
        self.CNN_most_recent = None
        self.CNN_most_recent_path = None
        self.CNN_history = []

        self.num_VAE_epochs = 0
        self.num_CNN_epochs = 0

        self.device = device
        self.VAE_optimizer = None
        self.CNN_optimizer = None

        self.overall_average_reconstruction_error = None
        self.queue_length = 9999
        self.history_queue_of_task_relatedness = []
        self.is_need_to_recalibrate_queue = False
        self.queue_sum = 0
        self.task_relatedness_threshold = 0.95


        self.sample_limit = float('Inf')

        self.dataset_splits = self.dataset_interface.dataset_splits
        self.by_category_record_of_reconstruction_error = None
        self.by_category_mean_std_of_reconstruction_error = None
        self.has_VAE_changed_since_last_mean_std_measurements = True

        self.mutated_count = 0
        self.fixed_noise = self.dataset_interface.list_of_fixed_noise

        self.record_keeper = record_keeper



    def refresh_variables_for_mutated_task(self):
        self.mutated_count += 1
        self.num_categories_in_task = self.dataset_interface.num_categories
        print("num cat 2",self.num_categories_in_task)
        self.categories_list = self.dataset_interface.categories_list
        self.by_category_record_of_reconstruction_error = None
        self.by_category_mean_std_of_reconstruction_error = None


    def create_and_train_VAE(self, model_id, num_epochs=30, batch_size=64, hidden_dim=10, latent_dim=50,
                             is_synthetic=False, is_take_existing_VAE=False, teacher_VAE=None,
                             is_new_categories_to_addded_to_existing_task= False, is_completely_new_task=False,
                             epoch_improvement_limit=30, learning_rate=0.00035, betas=(0.5, .999),weight_decay = 0.0001,sample_limit= float('Inf'), is_save=False, ):

        self.num_VAE_epochs = num_epochs

        # if not using a pre-trained VAE, create new VAE
        if not is_take_existing_VAE:
            # need to fix hidden dimensions
            self.VAE_most_recent = CVAE.CVAE(self.dataset_interface.original_input_dimensions, hidden_dim, latent_dim,
                                             self.num_categories_in_task,
                                             self.dataset_interface.original_channel_number, self.device)
            print("create new VAE to handle ",self.num_categories_in_task, self.categories_list)
        # if using a pre-trained VAE
        else:
            self.VAE_most_recent = copy.deepcopy(teacher_VAE)

            # if adding a new task category to learn
            if (is_completely_new_task or is_new_categories_to_addded_to_existing_task):
                fc3 = nn.Linear(self.num_categories_in_task, 1000)
                fc2 = nn.Linear(self.num_categories_in_task, 1000)

                # update CVAE class parameters
                self.VAE_most_recent.update_y_layers(fc3, fc2)
                self.VAE_most_recent.n_categories = self.num_categories_in_task

            self.VAE_most_recent.to(self.device)

        patience_counter = 0
        best_train_loss = 100000000000
        self.VAE_optimizer = torch.optim.Adam(self.VAE_most_recent.parameters(), lr=learning_rate, betas=betas,weight_decay=weight_decay)

        #if (not is_new_categories_to_addded_to_existing_task):
        dataloaders = self.dataset_interface.return_data_loaders('VAE', BATCH_SIZE=batch_size)
        #else:
         #   dataloaders = self.dataset_interface.obtain_dataloader_with_subset_of_categories('VAE',
          #                                                                                   new_categories_to_add_to_existing_task,
           #                                                                                  BATCH_SIZE=batch_size)

        start_timer = time.time()
        for epoch in range(num_epochs):

            # fix epochs
            train_loss = self.run_a_VAE_epoch_calculate_loss(dataloaders['train'], is_train=True,
                                                             is_synthetic=is_synthetic, sample_limit=sample_limit)
            val_loss = self.run_a_VAE_epoch_calculate_loss(dataloaders['val'], is_train=False,
                                                           is_synthetic=is_synthetic)

            if sample_limit < float('inf'):
                train_size = sample_limit
            else:
                train_size =self.dataset_interface.training_set_size

            train_loss /= train_size
            val_loss /= self.dataset_interface.val_set_size
            time_elapsed = time.time() - start_timer

            if best_train_loss > train_loss:
                best_train_loss = train_loss
                patience_counter = 1
                best_model_wts = copy.deepcopy(self.VAE_most_recent.state_dict())
                # send to CPU to save on GPU RAM

                print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f} ', time_elapsed,
                      "**** new best ****")


            else:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f} ', time_elapsed)
                patience_counter += 1

            if patience_counter > epoch_improvement_limit:
                break

        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best train loss: {:4f}'.format(best_train_loss))

        self.VAE_most_recent.load_state_dict(best_model_wts)

        model_string = "VAE " + self.task_name + " epochs" + str(num_epochs) + ",batch" + str(
            batch_size) + ",z_d" + str(
            latent_dim) + ",synth" + str(is_synthetic) + ",rebuilt" + str(is_take_existing_VAE) + ",lr" + str(
            learning_rate) + ",betas" + str(betas) + "lowest_error " + str(best_train_loss) + " "
        model_string += model_id

        self.VAE_most_recent_path = self.initial_directory_path + model_string
        self.overall_average_reconstruction_error = best_train_loss
        self.VAE_most_recent.cpu()

        # reset related task queue variables
        self.reset_queue_variables()

        if is_save:
            torch.save(self.VAE_most_recent, self.VAE_most_recent_path)


        #REMOVE COMMENT
       #self.obtain_VAE_recon_error_mean_and_std_per_class(self.initial_directory_path + model_string)

        self.VAE_most_recent.send_all_to_CPU()
        torch.cuda.empty_cache()

    def reset_queue_variables(self):
        self.history_queue_of_task_relatedness = []
        self.is_need_to_recalibrate_queue = False
        self.queue_sum = 0

    def run_a_VAE_epoch_calculate_loss(self, data_loader, is_train, teacher_VAE=None, is_synthetic=False,
                                       is_cat_info_required=False, sample_limit = float('Inf')):

        #self.VAE_most_recent.send_all_to_GPU()
        if is_train:
            self.VAE_most_recent.train()
        else:
            self.VAE_most_recent.eval()

        # To update record for mean and std deviation distance. This deletes old entries before calculating
        if is_cat_info_required:
            self.by_category_record_of_reconstruction_error = {i: [] for i in range(0, self.num_categories_in_task)}

        loss_sum = 0
        for i, (x, y) in enumerate(data_loader):

            # To restrict training size
            if sample_limit < (i*data_loader.batch_size):
                #print("break at ", i*data_loader.batch_size)
                break
            # reshape the data into [batch_size, 784]
            #print(x.size())
            #x = x.view(batch_size, 1, 28, 28)
            x = x.to(self.device)
            #print(y)

            if is_cat_info_required:
                cat_record = self.by_category_record_of_reconstruction_error[y.item()]

            # convert y into one-hot encoding
            y_original = y
            #print("going into one hot")
            # print(y)
            y = Utils.idx2onehot(y.view(-1, 1), self.num_categories_in_task)
            y = y.to(self.device)


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
            # print("train? ",is_train, i)
            # print(loss.item())
            # print(loss)

            if is_cat_info_required:
                cat_record.append(loss.item())

            if (is_train):
                loss.backward()
                self.VAE_optimizer.step()
                self.VAE_optimizer.zero_grad()

            del x,y,loss,reconstructed_x, z_mu, z_var
            torch.cuda.empty_cache()



        #self.VAE_most_recent.send_all_to_CPU()
        return loss_sum

    # def generate_synthetic_batch(self, batch_size = 64):

    def load_existing_VAE(self, PATH, is_calculate_mean_std=False):
        self.VAE_most_recent = torch.load(PATH)

        #REMOVE COMMENTS!!!!!
        if is_calculate_mean_std:
            print(self.task_name, " - Loaded VAE model, now calculating reconstruction error mean, std")
            self.obtain_VAE_recon_error_mean_and_std_per_class(PATH)
        else:
            self.by_category_mean_std_of_reconstruction_error = mpu.io.read(PATH + "mean,std.pickle")
            self.overall_average_reconstruction_error = mpu.io.read(PATH + "overallmean.pickle")

    def create_and_train_CNN(self, model_id, num_epochs=30, batch_size=64, is_frozen=False, is_off_shelf_model=False,
                             epoch_improvement_limit=20, learning_rate=0.0003, betas=(0.999, .999), weight_decay = 0.0001, is_save=False,sample_limit = float('Inf'), is_synthetic_blend = False):


        self.num_CNN_epochs = num_epochs


        # need to fix hidden dimensions

        if is_off_shelf_model:
            self.CNN_most_recent = models.resnet18(pretrained=True)

            if is_frozen == True:
                for param in self.CNN_most_recent.parameters():
                    param.requires_grad = False

            # take number of features from last layer
            num_ftrs = self.CNN_most_recent.fc.in_features

            # create a new fully connected final layer for training
            #print("self.num_categories_in_task", self.num_categories_in_task)
            self.CNN_most_recent.fc = nn.Sequential(nn.Linear(num_ftrs, self.num_categories_in_task),nn.LogSoftmax(dim=1))

            if is_frozen == True:
                # only fc layer being optimised
                self.CNN_optimizer = torch.optim.Adam(self.CNN_most_recent.fc.parameters(), lr=learning_rate,
                                                      betas=betas, weight_decay=weight_decay)
            else:
                # all layers being optimised
                self.CNN_optimizer = torch.optim.Adam(self.CNN_most_recent.parameters(), lr=learning_rate, betas=betas,weight_decay=weight_decay)

        else:
            self.CNN_most_recent = None
            # all layers being optimised
            self.CNN_optimizer = torch.optim.Adam(self.CNN_most_recent.parameters(), lr=learning_rate, betas=betas,weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()

        dataloaders = self.dataset_interface.return_data_loaders('CNN', BATCH_SIZE=batch_size)

        start_timer = time.time()
        best_train_acc = 0.0
        self.CNN_most_recent.to(self.device)
        patience_counter = 0

        for epoch in range(num_epochs):

            train_loss, correct_guesses_train = self.run_a_CNN_epoch_calculate_loss(dataloaders['train'], criterion,sample_limit = sample_limit,
                                                                                    is_train=True)
            val_loss, correct_guesses_val = self.run_a_CNN_epoch_calculate_loss(dataloaders['val'], criterion,
                                                                                is_train=False)

            train_loss /= self.dataset_interface.training_set_size
            val_loss /= self.dataset_interface.val_set_size

            train_acc = correct_guesses_train / self.dataset_interface.training_set_size
            val_acc = correct_guesses_val / self.dataset_interface.val_set_size

            time_elapsed = time.time() - start_timer

            if best_train_acc < train_acc:
                best_train_acc = train_acc
                patience_counter = 1

                # send to CPU to save on GPU RAM
                best_model_wts = copy.deepcopy(self.CNN_most_recent.state_dict())
                print(f'Epoch {epoch}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} ', time_elapsed,
                      "**** new best ****",correct_guesses_train.item(),"out of",self.dataset_interface.training_set_size,"out of",correct_guesses_val.item(),self.dataset_interface.val_set_size)


            else:
                print(f'Epoch {epoch}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} ', time_elapsed,correct_guesses_train,self.dataset_interface.training_set_size,correct_guesses_val,self.dataset_interface.val_set_size)
                patience_counter += 1

            if patience_counter > epoch_improvement_limit:
                break

        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}'.format(best_train_acc))

        self.CNN_most_recent.load_state_dict(best_model_wts)

        model_string = "CNN " + self.task_name + " epochs" + str(num_epochs) + ",batch" + str(
            batch_size) + ",pretrained" + str(is_off_shelf_model) + ",frozen" + str(is_frozen) + ",lr" + str(
            learning_rate) + ",betas" + str(betas) + " accuracy " + str(best_train_acc.item()) + " "
        model_string += model_id

        self.CNN_most_recent_path = self.initial_directory_path + model_string

        if is_save:
            torch.save(self.CNN_most_recent, self.initial_directory_path + model_string)

        self.CNN_most_recent.cpu()
        torch.cuda.empty_cache()

    def obtain_VAE_recon_error_mean_and_std_per_class(self, VAE_MODEL_PATH=None):

        print("...updating VAE record means, standard deviations...")

        data_loader = self.dataset_interface.return_data_loaders(branch_component='VAE',
                                                                 BATCH_SIZE=1)

        # get a record of loss reconstruction on training set, but don't do any training
        self.VAE_most_recent.send_all_to_GPU()
        train_loss = self.run_a_VAE_epoch_calculate_loss(data_loader['train'], is_train=False, is_cat_info_required=True)
        self.VAE_most_recent.send_all_to_CPU()
        self.overall_average_reconstruction_error = train_loss/self.dataset_interface.training_set_size

        # initialize dictionary
        self.by_category_mean_std_of_reconstruction_error = {}

        for cat in range(0, self.num_categories_in_task):
            mean = np.asarray(self.by_category_record_of_reconstruction_error[cat]).mean(axis=0)
            std = np.asarray(self.by_category_record_of_reconstruction_error[cat]).std(axis=0)
            self.by_category_mean_std_of_reconstruction_error[cat] = (mean, std)

        mpu.io.write(VAE_MODEL_PATH + "mean,std.pickle", self.by_category_mean_std_of_reconstruction_error)
        mpu.io.write(VAE_MODEL_PATH + "overallmean.pickle", self.overall_average_reconstruction_error)



    # def

    def given_observation_find_lowest_reconstruction_error(self, x, is_standardised_distance_check=True):

        lowest_recon_error, recon_x = self.VAE_most_recent.get_sample_reconstruction_error_from_all_category(x,
                                                                                      self.by_category_mean_std_of_reconstruction_error,
                                                                                      is_random=False,
                                                                                      only_return_best=True,
                                                                                      is_standardised_distance_check=is_standardised_distance_check)
        if len(self.history_queue_of_task_relatedness) >= self.queue_length:
            old_val = self.history_queue_of_task_relatedness.pop(0)
            self.queue_sum += lowest_recon_error[0][1] - old_val
            self.history_queue_of_task_relatedness.append(lowest_recon_error[0][1])
            average_recon = self.queue_sum/self.queue_length
          #  print(self.queue_sum , average_recon)
            average_task_relatedness = self.given_average_recon_error_calc_task_relatedness(average_recon)
          #  print("average_task_relatedness",average_task_relatedness)
            if average_task_relatedness < self.task_relatedness_threshold:
              #  print("reached 2")
                self.is_need_to_recalibrate_queue = True
        else:
            #print("reached 3")
            self.history_queue_of_task_relatedness.append(lowest_recon_error[0][1])
            self.queue_sum += lowest_recon_error[0][1]


        return lowest_recon_error, recon_x


    def given_task_data_set_find_task_relatedness(self, new_task_data_loader, num_samples_to_check,shear_degree = 0):

        reconstruction_error_sum = 0
        sample_count = 0
        for i, (x, y) in enumerate(new_task_data_loader):

            x = x.float()

            # if shear_degree != -1:
            #     #print(x.size())
            #     x = torch.squeeze(x)
            #     #print(x.size())
            #     shear_trans = transforms.Compose([transforms.ToPILImage(),lambda img: transforms.functional.affine(img, angle=0,translate=(0,0), scale=1, shear=shear_degree),transforms.ToTensor()])
            #     x = shear_trans(x).float()

            lowest_recon_error, recon_x = self.given_observation_find_lowest_reconstruction_error(x,False)
            reconstruction_error_sum += lowest_recon_error[0][1]

            if i > num_samples_to_check:
                break

            sample_count = i

        reconstruction_error_average = reconstruction_error_sum/sample_count

        task_relatedness = self.given_average_recon_error_calc_task_relatedness(reconstruction_error_average)

        return reconstruction_error_average, task_relatedness

    def given_average_recon_error_calc_task_relatedness(self, reconstruction_error_average):
        task_relatedness = 1 - abs(self.overall_average_reconstruction_error - reconstruction_error_average) / (
                    self.overall_average_reconstruction_error + reconstruction_error_average)
        return task_relatedness

    def generate_single_random_sample(self, category, is_random_cat=False):
        return self.VAE_most_recent.generate_single_random_sample(category, is_random_cat)



    def run_a_CNN_epoch_calculate_loss(self, data_loader, criterion, is_train, samples_limited_percentage = 1, sample_limit = float('Inf'),is_synthetic_blend = False):

        running_loss = 0.0
        running_corrects = 0

        if is_train:
            self.CNN_most_recent.train()
        else:
            self.CNN_most_recent.eval()



        loss_sum = 0.0
        running_corrects = 0
        for i, (x, y) in enumerate(data_loader):

            if sample_limit < (i * data_loader.batch_size):
                break
            #if(samples_limited_percentage*data_loader.batch)

            # reshape the data into [batch_size, 784]
            #print(x.size())
            # x = x.view(batch_size, 1, 28, 28)
            #   print(y)
            x = x.to(self.device)
            y = y.to(self.device)
            #print(y.size())
            #print(y)

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
                #print(preds)
                loss = criterion(outputs.to(self.device), y)
                #print(loss)

            loss_sum += loss.item()
            running_corrects += torch.sum(preds == y.data)

            # loss
            # print("train? ",is_train, i)
            # print(preds)
            # print(preds == y.data)
            # print(torch.sum(preds == y.data))
            # print(running_corrects)
            # if(running_corrects> (i+1)*64):
            #     print("EXCEEDED\n\n\n\n")

            # backward + optimize only if in training phase
            if is_train:
                loss.backward()
                self.CNN_optimizer.step()

            del x,y,loss,preds,outputs
            torch.cuda.empty_cache()

            # To restrict training size


        return loss_sum, running_corrects.double()

    def load_existing_CNN(self, PATH):
        self.CNN_most_recent = torch.load(PATH)

    def classify_with_CNN(self, x, is_output_one_hot=False):

        self.CNN_most_recent.to(self.device)
        x = x.to(self.device)
        output = self.CNN_most_recent(x)
        preds = output

        #print(output)

        if not is_output_one_hot:
            max_value, preds = torch.max(output, 1)


        return preds, max_value

    def create_blended_dataset_with_synthetic_samples(self, new_real_datasetInterface,new_categories_to_add_to_existing_task,extra_new_cat_multi=1, is_single_task_recal_process= False):

        is_plot = False

        if len(new_categories_to_add_to_existing_task) ==0:
            new_categories_to_add_to_existing_task = new_real_datasetInterface.categories_list

        synthetic_cat_number = len(self.categories_list)
        if not is_single_task_recal_process:
                self.categories_list.extend(new_categories_to_add_to_existing_task)
        print(self.categories_list)
        self.synthetic_samples_for_reuse = {i: [] for i in self.dataset_splits}

        original_cat_index_to_new_cat_index_dict = {new_real_datasetInterface.label_to_index_dict[new_cat_label]:self.categories_list.index(new_cat_label) for new_cat_label in new_categories_to_add_to_existing_task}
        print(original_cat_index_to_new_cat_index_dict )

        blended_dataset = {split: {} for split in self.dataset_splits}

        real_db = None

        for model in ['VAE', 'CNN']:

            for split in self.dataset_splits:

                #subset_dataset_real = new_real_datasetInterface.obtain_dataset_with_subset_of_categories(model,split,new_categories_to_add_to_existing_task)
                # always VAE for synthetic
                subset_dataset_real = new_real_datasetInterface.obtain_dataset_with_subset_of_categories('VAE',split,new_categories_to_add_to_existing_task)

                real_db = subset_dataset_real
               # print(len(real_db))

                size_per_class = 4
                size_per_class = round(len(subset_dataset_real) / (len(new_categories_to_add_to_existing_task)*(1/extra_new_cat_multi)))
                self.synthetic_samples_for_reuse[split] = []


                _, _ = self.VAE_most_recent.generate_synthetic_set_all_cats(
                    synthetic_data_list_unique_label=self.synthetic_samples_for_reuse[split],
                    number_per_category=size_per_class, is_store_on_CPU = True)
                print("SYN AND REAL model",model,"split",split, "size per class", size_per_class)


                blended_dataset[split][model] = CustomDataSetAndLoaderForSynthetic.SyntheticDS(
                    self.synthetic_samples_for_reuse[split], self.dataset_interface.transformations,subset_dataset_real,self.categories_list, synthetic_cat_number, original_cat_index_to_new_cat_index_dict)



        self.dataset_interface.update_data_set(blended_dataset)
        self.refresh_variables_for_mutated_task()

        if is_plot:
            fig1 = plt.figure(figsize=(10, 10))
            x = 0
            r = 20
            c = 3

            for i in range(x,r*c):
                img, cat = self.dataset_interface.dataset['val']['VAE'][i]

                img = img.view(28, 28).data
                img = img.numpy()
                ax = fig1.add_subplot(r, c, i-x + 1)
                ax.axis('off')
                ax.set_title(cat.item())
                ax.imshow(img, cmap='gray_r')

            plt.ioff()
            plt.show()

        # fig2 = plt.figure(figsize=(10, 10))
        # x = 30
        # r = 10
        # c = 3
        #
        # for i in range(x, r * c):
        #     print(len(self.dataset_interface.dataset['train']['VAE']))
        #     img, cat = self.dataset_interface.dataset['train']['VAE'][i]
        #     img = img.view(28, 28).data
        #     img = img.numpy()
        #     ax = fig2.add_subplot(r, c, i - x + 1)
        #     ax.axis('off')
        #     ax.set_title(cat)
        #     ax.imshow(img, cmap='gray_r')
        #
        # plt.ioff()
        # plt.show()

    def run_end_of_training_benchmarks(self, test_name, dataset_interface = None, is_save = True, is_gaussian_noise_required = False, is_extra_top_three_method = False):

        is_plot_results = False

        if dataset_interface == None:
            dataset_interface = self.dataset_interface

        print("\n *************\nFor task ", self.task_name)
        # print(self.VAE_most_recent_path)
        # print(self.CNN_most_recent_path)

        data_loader_CNN = dataset_interface.return_data_loaders('CNN', BATCH_SIZE=1)
        data_loader_VAE = dataset_interface.return_data_loaders('VAE', BATCH_SIZE=1)
        data_loader_VAE2 = dataset_interface.return_data_loaders('VAE', BATCH_SIZE=1)

        self.VAE_most_recent.eval()
        self.CNN_most_recent.eval()
        self.VAE_most_recent.send_all_to_GPU()
        self.CNN_most_recent.to(self.device)

        # To update record for mean and std deviation distance. This deletes old entries before calculating
        by_category_record_of_recon_error_and_accuracy = {i: [0, 0, 0] for i in
                                                          range(0, self.num_categories_in_task)}

        for i, (x, y) in enumerate(data_loader_VAE['val']):
            x = x.to(self.device)
            y_original = y.item()
            y = Utils.idx2onehot(y.view(-1, 1), self.num_categories_in_task)
            y = y.to(self.device)

            with torch.no_grad():
                reconstructed_x, z_mu, z_var = self.VAE_most_recent(x, y)

            # loss
            loss = self.VAE_most_recent.loss(x, reconstructed_x, z_mu, z_var)

            by_category_record_of_recon_error_and_accuracy[y_original][0] += 1
            by_category_record_of_recon_error_and_accuracy[y_original][1] += loss.item()

        classified_nines = 0

        if not is_gaussian_noise_required:
            for i, (x, y) in enumerate(data_loader_CNN['val']):
                # reshape the data into [batch_size, 784]
                # print(x.size())
                # x = x.view(batch_size, 1, 28, 28)
                x = x.to(self.device)
                y_original = y.item()
                y = y.to(self.device)


                with torch.no_grad():
                    outputs = self.CNN_most_recent(x)
                    prob, preds = torch.max(outputs, 1)
                    # print(preds)
                    # criterion = nn.CrossEntropyLoss()
                    # loss = criterion(outputs.to(device), y)

                    correct = torch.sum(preds == y.data)

                by_category_record_of_recon_error_and_accuracy[y_original][2] += correct.item()

                if preds.item() == self.num_categories_in_task-1:
                    classified_nines += 1

                if is_plot_results:
                    if(i<10):
                        print("actual",y.data,"classified",preds, "no Gaussian noise")
                        fig1 = plt.figure(figsize=(5, 5))
                        ax = fig1.add_subplot(1, 3, 1)
                        ax.axis('off')
        #                generated_x, _, _ = self.VAE_most_recent.encode_then_decode_without_randomness(x,y)
                        ax.imshow(x[0][0].cpu().numpy(), cmap='gray')

                        plt.ioff()
                        plt.show()

        else:
            for i, (x, y) in enumerate(data_loader_VAE2['val']):
                # reshape the data into [batch_size, 784]
                # print(x.size())
                # x = x.view(batch_size, 1, 28, 28)
                x_noisy = gaussian_filter(x.cpu().detach().numpy(), sigma=.5)
                #x_noisy = gaussian_filter(x.cpu().detach().numpy(), sigma=1)
                x_noisy = torch.Tensor(x_noisy)
                x_original = x.to(self.device)
                y_original = y
                y_one_hot = Utils.idx2onehot(y.view(-1, 1), self.num_categories_in_task)
                y = y.to(self.device)
                y_one_hot = y_one_hot.to(self.device)

                #generated_x, _, _ = self.VAE_most_recent.encode_then_decode_without_randomness(x,y_one_hot.float())
                #__, generated_x = self.VAE_most_recent.get_sample_reconstruction_error_from_all_category(x)
                #print(x_noisy.size())
                x_trans = self.dataset_interface.transformations['CNN']['test_to_image']( torch.squeeze(x_noisy))
                #x = self.dataset_interface.transformations['CNN']['test_to_image'](torch.squeeze(generated_x.cpu()).detach().numpy())
                x = torch.unsqueeze(x_trans,0)
                #print(generated_x.size())
                #print(x.size())
                x = x.to(self.device)


                with torch.no_grad():
                    outputs = self.CNN_most_recent(x)

                    prob, preds = torch.max(outputs, 1)

                    # print(outputs)
                    # print(preds)
                    # print(prob)
                    # print(y.data)
                    n_checks = 10


                    if (prob<math.log(0.999999999) and is_extra_top_three_method):

                        one_hot_list = []
                        index_to_cat = []
                        prob_list = []
                        idx = (-outputs.cpu().numpy()).argsort()[:n_checks]
                        # print("here", math.log(0.8))
                        # print("idx", idx)
                        for i in range(0,n_checks):
                            y_one_hot1 = Utils.idx2onehot(torch.tensor([[idx[0,min(i,len(idx))]]]), self.num_categories_in_task).to('cuda')
                            #y_one_hot3 = Utils.idx2onehot(torch.tensor([[idx[0,min(2,len(idx))]]]), self.num_categories_in_task).to('cuda')
                            #y_one_hot2 = Utils.idx2onehot(torch.tensor([[idx[0,min(i,len(idx))]]]), self.num_categories_in_task).to('cuda')

                            x_recon1, _, _ = self.VAE_most_recent.encode_then_decode_without_randomness(x_original, y_one_hot)
                            #x_recon2, _, _ = self.VAE_most_recent.encode_then_decode_without_randomness(x_original, y_one_hot2)
                            #x_recon3, _, _ = self.VAE_most_recent.encode_then_decode_without_randomness(x_original, y_one_hot3)
                            x_trans1 = self.dataset_interface.transformations['CNN']['test_to_image'](torch.squeeze(x_recon1.cpu()))
                            #x_trans2 = self.dataset_interface.transformations['CNN']['test_to_image'](torch.squeeze(x_recon2.cpu()))
                            #x_trans3 = self.dataset_interface.transformations['CNN']['test_to_image'](torch.squeeze(x_recon3.cpu()))

                            outputs1 = self.CNN_most_recent(torch.unsqueeze(x_trans1,0).to(self.device))
                            prob1, preds1 = torch.max(outputs1, 1)
                            prob_list.append(prob1)
                            index_to_cat.append(preds1)

                        #outputs2 = self.CNN_most_recent(torch.unsqueeze(x_trans2,0).to(self.device))
                        #prob2, preds2 = torch.max(outputs2, 1)
                        #outputs3 = self.CNN_most_recent(torch.unsqueeze(x_trans3,0).to(self.device))
                        #prob3, preds3 = torch.max(outputs3, 1)
                        #index_to_cat = [preds1,preds2,preds3]
                        # print(prob_ten)
                        #print(prob_list)
                        prob_ten = torch.tensor([prob_list])
                        prob, preds = torch.max(prob_ten,1)
                        cat_chosen = index_to_cat[preds.cpu().item()]
                        preds =cat_chosen


                    # print(preds)
                    # print(prob)
                    # print(y.data,"\n")

                    correct = torch.sum(preds == y.data)

                by_category_record_of_recon_error_and_accuracy[y_original.item()][2] += correct.item()

                if preds.item() == 9:
                    classified_nines += 1

                if is_plot_results:
                    if(i<20):
                        print("actual",y.data,"classified",preds, "contains gaussian noise")
                        fig1 = plt.figure(figsize=(5, 5))
                        ax = fig1.add_subplot(1, 3, 1)
                        ax.axis('off')
                       # generated_x, _, _ = self.VAE_most_recent.encode_then_decode_without_randomness(x,y)
                        ax.imshow(x[0][0].cpu().numpy(), cmap='gray')

                        plt.ioff()
                        plt.show()

        self.CNN_most_recent.cpu()
        self.VAE_most_recent.send_all_to_CPU()


        total_count = 0
        total_recon = 0
        total_correct = 0

        recon_per_class = [0 for i in range(0, self.num_categories_in_task)]
        accuracy_per_class = [0 for i in range(0, self.num_categories_in_task)]

        for category in by_category_record_of_recon_error_and_accuracy:
            count = by_category_record_of_recon_error_and_accuracy[category][0]
            recon_ave = by_category_record_of_recon_error_and_accuracy[category][1] / count
            accuracy = by_category_record_of_recon_error_and_accuracy[category][2] / count

            total_count += by_category_record_of_recon_error_and_accuracy[category][0]
            total_recon += by_category_record_of_recon_error_and_accuracy[category][1]
            total_correct += by_category_record_of_recon_error_and_accuracy[category][2]

            recon_per_class[category] +=recon_ave
            accuracy_per_class[category] +=accuracy

            print("For:", count, category, " Ave. Recon:", recon_ave, " Ave. Accuracy:", accuracy)

        total_average_recon = total_recon / total_count
        total_accuracy = total_correct / total_count
        print("For all (", total_count, "): Ave. Recon:", total_average_recon, " Ave. Accuracy:", total_accuracy,"\n")
        print(self.num_categories_in_task-1,"classifications",classified_nines)

        if is_save:
            epoch_record = EpochRecord.EpochRecord(test_name, self.task_name, str(self.num_VAE_epochs), str(self.num_CNN_epochs), total_count, self.num_categories_in_task, total_average_recon,total_accuracy, self.categories_list, recon_per_class, accuracy_per_class, random_image_per_class_list=None)

            self.record_keeper.record_iteration_accuracy(test_name, epoch_record)

    def generate_samples_to_display(self):

        self.VAE_most_recent.eval()
        self.VAE_most_recent.to(self.device)
        z1 = self.fixed_noise[0]
        z2 = self.fixed_noise[1]
        z3 = self.fixed_noise[2]

        r = self.num_categories_in_task
        c = 1
        #c = round(self.num_categories_in_task/2)

        for cat in range(0,self.num_categories_in_task):

            fig1 = plt.figure(figsize=(5, 5))
            #print(cat)
            img1 = self.VAE_most_recent.generate_single_random_sample(cat ,z1).cpu().numpy()
            img2 = self.VAE_most_recent.generate_single_random_sample(cat ,z2).cpu().numpy()
            #img2 = self.dataset_interface.transformations['CNN']['test_to_image'](self.VAE_most_recent.generate_single_random_sample(cat ,z2).cpu())[0].cpu().numpy()
            img3 = self.VAE_most_recent.generate_single_random_sample(cat ,z3).cpu().numpy()


            #ax = fig1.add_subplot(r, c, cat + 1)
            ax = fig1.add_subplot(1, 3, 1)
            ax.axis('off')
            #ax.set_title(cat)
            ax.imshow(img1, cmap='gray')

            ax = fig1.add_subplot(1, 3, 2)
            ax.axis('off')
            #ax.set_title(cat)
            ax.imshow(img2, cmap='gray')

            ax = fig1.add_subplot(1, 3, 3)
            ax.axis('off')
            #ax.set_title(cat)
            ax.imshow(img3, cmap='gray')

            plt.ioff()
            plt.show()

#        plt.ioff()
 #       plt.show()

        self.VAE_most_recent.cpu()
