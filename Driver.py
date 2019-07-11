import torch
import random
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import DatasetAndInterface as ds
import TaskBranch as task



manualSeed = random.randint(1, 10000)
manualSeed = 149
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


PATH_DATA_MNIST = "/cs/tmp/al278/MNIST"
PATH_DATA_FashionMNIST = "/cs/tmp/al278/FashionMNIST"
PATH_DATA_EMNIST = "/cs/tmp/al278/EMNIST"
PATH_MODELS = "/cs/tmp/al278/"

dataset_path_list = [(datasets.MNIST,PATH_DATA_MNIST),(datasets.FashionMNIST,PATH_DATA_FashionMNIST),(datasets.EMNIST,PATH_DATA_EMNIST)]

# as suggested by pytorch devs
normalise_for_PIL_mean = (0.5, 0.5, 0.5)
normalise_for_PIL_std = (0.5, 0.5, 0.5)
normalise_MNIST_mean = (0.1307,)
normalise_MNIST_std = (0.3081,)

image_height_MNIST = 28
image_channel_size_MNIST = 1

transforms_CNN_one_channel_to_three = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
    ]),
}

transforms_VAE_one_channel = {
    'train': transforms.Compose([
        transforms.Resize(image_height_MNIST ),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        #transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_height_MNIST ),
        transforms.ToTensor(),
        #transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_height_MNIST ),
        transforms.ToTensor(),
        #transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
    ]),
}

all_transforms = {}
all_transforms['VAE'] = transforms_VAE_one_channel
all_transforms['CNN'] = transforms_CNN_one_channel_to_three

for (dataset, dataset_path) in dataset_path_list:

    # from pytorch tutorial
    image_datasets_MNIST = {}
    image_datasets_FashionMNIST = {}
    image_datasets_EMNIST = {}
    for x in ['train', 'val']:
        image_datasets_MNIST[x] = {}
        image_datasets_FashionMNIST[x] = {}
        image_datasets_EMNIST[x] = {}
        for y in ['VAE','CNN']:
            image_datasets_MNIST[x][y] = datasets.MNIST(PATH_DATA_MNIST, train=(x == 'train'), download=True, transform = all_transforms[y][x])
            image_datasets_FashionMNIST[x][y] = datasets.FashionMNIST(PATH_DATA_FashionMNIST, train=(x == 'train'), download=True, transform = all_transforms[y][x])
            image_datasets_EMNIST[x][y] = datasets.EMNIST(PATH_DATA_EMNIST, train=(x == 'train'), download=True, transform = all_transforms[y][x], split='letters')


minist_data_and_interface = ds.DataSetAndInterface('MNIST', image_datasets_MNIST,PATH_DATA_MNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST)
fashion_minist_data_and_interface = ds.DataSetAndInterface('FashionMNIST', image_datasets_FashionMNIST,PATH_DATA_FashionMNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST)

mnist_task_branch = task.TaskBranch('MNIST', minist_data_and_interface, PATH_MODELS)

#mnist_task_branch.create_and_train_VAE()

fashion_mnist_task_branch = task.TaskBranch('Fashion', fashion_minist_data_and_interface, PATH_MODELS)

#fashion_mnist_task_branch.create_and_train_VAE()



latent_dim = [50, 75, 100]
learning_rate = [0.00025, 0.0003, 0.00035]
betas=[(0.5, .999),(0.75, .999),(0.999, .999)]

for ld in latent_dim:
    for lr in learning_rate:
        for b in betas:
            print("\n****************\nnew hyperparemeters  ")
            print("For MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b)
            mnist_task_branch.create_and_train_VAE(num_epochs=30, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=20, learning_rate=lr, betas=b)
            print("For MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b)


            print("\nFor Fashion MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b)
            fashion_mnist_task_branch.create_and_train_VAE(num_epochs=30, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=20, learning_rate=lr, betas=b)
            print("For Fashion MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b)

# establish classes
# load data



# initialisation training
#  - test accuracy of gate
#  - test accuracy with and without expert gate
#  - test generation-classification matches


# new task

# show gate failing
#  - adding category to existing task with transfer learning and student teacher psuedo-rehersal
#  - adding new category with transfer learning


# show gate failing
#  - test overall system accuracy relative to model from scratch
#  - test overall system accuracy relative to model from scratch

