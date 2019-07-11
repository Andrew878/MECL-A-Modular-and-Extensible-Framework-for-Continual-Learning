import torch
import torch.optim
import CVAE
import Utils
import time
import copy
from torchvision import models
import torch.nn as nn



class TaskBranch:

    def __init__(self, task_name, dataset_interface, initial_directory_path):

        self.task_name = task_name
        self.dataset_interface = dataset_interface
        self.initial_directory_path = initial_directory_path
        self.num_categories_in_task = self.dataset_interface.num_categories
        self.num_categories_in_task = self.dataset_interface.num_categories
        print(self.num_categories_in_task)

        self.VAE_most_recent = None
        self.VAE_history = []
        self.CNN_most_recent = None
        self.CNN_history = []

        self.device = 'cuda'
        self.VAE_optimizer = None
        self.CNN_optimizer = None

        self.synthetic_samples_for_reuse =[]

    def create_and_train_VAE(self, num_epochs=30, batch_size = 64, hidden_dim=10, latent_dim=75, is_synthetic=False, is_take_existing_VAE=False, teacher_VAE = None, new_categories_to_add=0,
                             epoch_improvement_limit=20, learning_rate=0.0003, betas=(0.5, .999)):

        self.synthetic_samples_for_reuse = []

        # if not using a pre-trained VAE
        if not is_take_existing_VAE:
            # need to fix hidden dimensions
            self.VAE_most_recent = CVAE.CVAE(self.dataset_interface.original_input_dimensions, hidden_dim, latent_dim,
                                         self.num_categories_in_task, self.dataset_interface.original_channel_number, self.device)

        # if using a pre-trained VAE
        else:
            self.VAE_most_recent = copy.deepcopy(teacher_VAE).to(self.device)

            # if adding a new task category to learn
            if (new_categories_to_add != 0):
                fc3 = nn.Linear(self.num_categories_in_task + new_categories_to_add, 1000)
                fc2 = nn.Linear(self.num_categories_in_task + new_categories_to_add, 1000)
                self.VAE_most_recent.update_y_layers(fc3, fc2)
                self.VAE_most_recent.to(self.device)


        patience_counter = 0
        best_val_loss = 100000000000
        self.VAE_optimizer = torch.optim.Adam(self.VAE_most_recent.parameters(), lr=learning_rate, betas=betas)

        dataloaders = self.dataset_interface.return_data_loaders('VAE', BATCH_SIZE=batch_size)

        start_timer = time.time()
        for epoch in range(num_epochs):

            train_loss = self.run_a_VAE_epoch_calculate_loss(dataloaders['train'], is_train=True,is_synthetic = is_synthetic)
            val_loss = self.run_a_VAE_epoch_calculate_loss(dataloaders['val'], is_train=False, is_synthetic = is_synthetic)

            train_loss /= self.dataset_interface.training_set_size
            val_loss /= self.dataset_interface.val_set_size

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                patience_counter = 1

                #send to CPU to save on GPU RAM
                best_model_wts = copy.deepcopy(self.VAE_most_recent.state_dict())
                print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f} ', time.time(),
                      "**** new best ****")


            else:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f} ', time.time())
                patience_counter += 1

            if patience_counter > epoch_improvement_limit:
                break

        time_elapsed = time.time() - start_timer
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}\n'.format(best_val_loss))

        model_string = "VAE_epochs"+str(num_epochs)+",batch"+str(batch_size)+",z_d"+str(latent_dim)+",synth"+ str(is_synthetic)+",rebuilt"+ str(is_take_existing_VAE)+",lr"+str(learning_rate)+",betas"+str(betas)
        model_string += " v1"

        self.VAE_most_recent.load_state_dict(best_model_wts)
        torch.save(self.VAE_most_recent,self.initial_directory_path+model_string)

        del self.VAE_most_recent
        torch.cuda.empty_cache()


    def run_a_VAE_epoch_calculate_loss(self, data_loader,is_train, teacher_VAE = None, is_synthetic = False):

        if is_train:
            self.VAE_most_recent.train()
        else:
            self.VAE_most_recent.eval()

        loss_sum = 0
        for i, (x, y) in enumerate(data_loader):
            # reshape the data into [batch_size, 784]
            #print(x.size())
            # x = x.view(batch_size, 1, 28, 28)
            x = x.to(self.device)
           # print(x)

            # convert y into one-hot encoding
            y = Utils.idx2onehot(y.view(-1, 1), self.num_categories_in_task)
            y = y.to(self.device)

            # get synthetic samples, and blend them into the batch
            if is_synthetic:
                synthetic_samples_x, synthetic_samples_y = teacher_VAE.generate_synthetic_set_all_cats(number_per_category=data_loader.batch_size)
                self.synthetic_samples_for_resuse.append((synthetic_samples_x, synthetic_samples_y))

                synthetic_samples_x.append(x)
                x = torch.cat((tuple(synthetic_samples_x)), dim=0)
                synthetic_samples_y.append(y)
                y = torch.cat((tuple(synthetic_samples_y)), dim=0)

            # update the gradients to zero
            self.VAE_optimizer.zero_grad()

            # forward pass
            # track history if only in train
            with torch.set_grad_enabled(is_train):
                reconstructed_x, z_mu, z_var = self.VAE_most_recent(x, y)

            # loss
            loss = self.VAE_most_recent.loss(x, reconstructed_x, z_mu, z_var)
            loss_sum += loss.item()
            #print(loss.item())

            if(is_train):
                loss.backward()
                self.VAE_optimizer.step()


        return loss_sum


    #def generate_synthetic_batch(self, batch_size = 64):








    def load_existing_VAE(self, PATH):
        self.VAE_most_recent = torch.load(PATH)


    def create_and_train_CNN(self, num_epochs=30,  batch_size = 64, hidden_dim=10, latent_dim=75, is_frozen=False, is_off_shelf_model = False,
                             epoch_improvement_limit=20, learning_rate=0.0003, betas=(0.999, .999)):

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
                self.CNN_optimizer = torch.optim.Adam(self.CNN_most_recent.fc.parameters(), lr=learning_rate, betas=betas)
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

            train_loss, correct_guesses_train = self.run_a_CNN_epoch_calculate_loss(dataloaders['train'],criterion, is_train=True)
            val_loss, correct_guesses_val = self.run_a_CNN_epoch_calculate_loss(dataloaders['val'], criterion,is_train=False)

            train_loss /= self.dataset_interface.training_set_size
            val_loss /= self.dataset_interface.val_set_size

            train_acc = correct_guesses_train / self.dataset_interface.training_set_size
            val_acc = correct_guesses_val / self.dataset_interface.val_set_size

            if best_val_acc > val_acc:
                best_val_acc = val_acc
                patience_counter = 1

                #send to CPU to save on GPU RAM
                best_model_wts = copy.deepcopy(self.CNN_most_recent.state_dict())
                print(f'Epoch {epoch}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.4f} ', time.time(),
                      "**** new best ****")


            else:
                print(f'Epoch {epoch}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.4f} ', time.time())
                patience_counter += 1

            if patience_counter > epoch_improvement_limit:
                break

        time_elapsed = time.time() - start_timer
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}\n'.format(best_val_acc))

        model_string = "CNN "+self.task_name+" epochs" + str(num_epochs) + ",batch" + str(batch_size) +",pretrained" + str(is_off_shelf_model) + ",frozen"+str(is_frozen)+",lr" + str(
            learning_rate) + ",betas" + str(betas)
        model_string += " v1"

        self.CNN_most_recent.load_state_dict(best_model_wts)
        torch.save(self.CNN_most_recent, self.initial_directory_path + model_string)

        del self.CNN_most_recent
        torch.cuda.empty_cache()




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
            #print(x.size())
            # x = x.view(batch_size, 1, 28, 28)
            x = x.to(self.device)
            y = y.to(self.device)
            #print(x.size())

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
                loss = criterion(outputs.to(self.device), y.to(self.device))

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

    def classify_with_CNN(self, x, is_output_one_hot = False):
        output = self.CNN_most_recent(x)
        preds = output

        if not is_output_one_hot:
            max_value, preds = torch.max(output, 1)

        return preds