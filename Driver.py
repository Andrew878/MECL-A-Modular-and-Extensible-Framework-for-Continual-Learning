import torch
import random
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import DatasetAndInterface as ds
import TaskBranch as task
import Utils
import Gate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manualSeed = random.randint(1, 10000)
manualSeed = 149
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


PATH_ROOT = "/cs/scratch/al278/"
PATH_ROOT = "/cs/tmp/al278/"
PATH_DATA_MNIST = str(PATH_ROOT)+"MNIST"
PATH_DATA_FashionMNIST = str(PATH_ROOT) + "FashionMNIST"
PATH_DATA_EMNIST = str(PATH_ROOT) + "EMNIST"
PATH_MODELS = str(PATH_ROOT) + "proper/"

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
    'test_to_image': transforms.Compose([
        transforms.ToPILImage(),
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
            # EMNIST has a index system starting at 1. All others have index system starting at 0. This makes an adjustment
            image_datasets_EMNIST[x][y].targets = image_datasets_EMNIST[x][y].targets -1


mnist_data_and_interface = ds.DataSetAndInterface('MNIST', image_datasets_MNIST,PATH_DATA_MNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST)
fashion_mnist_data_and_interface = ds.DataSetAndInterface('Fashion', image_datasets_FashionMNIST,PATH_DATA_FashionMNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST)

emnist_data_and_interface = ds.DataSetAndInterface('EMNIST', image_datasets_EMNIST,PATH_DATA_EMNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST)

mnist_task_branch = task.TaskBranch('MNIST', mnist_data_and_interface, device,PATH_MODELS)
fashion_mnist_task_branch = task.TaskBranch('Fashion', fashion_mnist_data_and_interface, device, PATH_MODELS)
emnist_task_branch = task.TaskBranch('EMNIST', emnist_data_and_interface, device, PATH_MODELS)


is_saving = False
label = "v2"
is_grid_search = False
BATCH = 128
EPOCHS = 200
ld = 50
b = (0.5, 0.999)
lr = 0.00035

if is_saving:
    BATCH = 128


    EPOCHS = 50
    print("Training MNIST CNN")
    mnist_task_branch.create_and_train_CNN(model_id = "small temporary", num_epochs=EPOCHS, batch_size=BATCH,  is_frozen=False,
                                           is_off_shelf_model=True, epoch_improvement_limit=20, learning_rate=0.00025,
                                           betas=(0.999, .999), is_save=True)

    print("Training Fashion MNIST CNN")
    fashion_mnist_task_branch.create_and_train_CNN(model_id = "small temporary", num_epochs=EPOCHS, batch_size=BATCH,  is_frozen=False,
                                                   is_off_shelf_model=True, epoch_improvement_limit=20, learning_rate=0.00025,
                                                   betas=(0.999, .999), is_save=True)

    EPOCHS = 600
    ld = 50
    b = (0.5,0.999)
    lr = 0.00035

    print("Training MNIST VAE")
    mnist_task_branch.create_and_train_VAE(model_id = label, num_epochs=EPOCHS, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=50, learning_rate=lr, betas=b,is_save=True, batch_size=BATCH)

    print("Training Fashion MNIST VAE")
    fashion_mnist_task_branch.create_and_train_VAE(model_id = label, num_epochs=EPOCHS, hidden_dim=10, latent_dim=ld,  is_synthetic=False,epoch_improvement_limit=50, learning_rate=lr, betas=b, is_save=True,batch_size=BATCH)

else:

    is_update_mean_std = False
    fashion_mnist_task_branch.load_existing_VAE(PATH_MODELS+"VAE Fashion epochs400,batch128,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 233.7498068359375 v2", is_update_mean_std)
    mnist_task_branch.load_existing_VAE(PATH_MODELS+"VAE MNIST epochs400,batch128,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 88.00115422363281 v2",is_update_mean_std)

    mnist_task_branch.load_existing_CNN(
        PATH_MODELS + "CNN MNIST epochs50,batch128,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) v1")
    fashion_mnist_task_branch.load_existing_CNN(
        PATH_MODELS + "CNN Fashion epochs50,batch128,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) v1")


task_branch_list = [mnist_task_branch,fashion_mnist_task_branch]
gate = Gate.Gate()



Utils.test_synthetic_samples_versus_normal(mnist_data_and_interface, emnist_data_and_interface, PATH_MODELS,device)


# i =1
# for list in [['a','b'],['c','d'],['e','f'],['g','h'],['i','j']]:
#
#     mnist_task_branch.create_blended_dataset_with_synthetic_samples(emnist_data_and_interface,list)
#     print("\nTraining VAE for ",list)
#     name= "mutation"+str(i)
#     mnist_task_branch.create_and_train_VAE( model_id=name, num_epochs=30, batch_size=64, hidden_dim=10, latent_dim=50,
#                                  is_synthetic=False, is_take_existing_VAE=True, teacher_VAE=mnist_task_branch.VAE_most_recent,
#                                  is_new_categories_to_addded_to_existing_task=True, is_completely_new_task=False,
#                                  epoch_improvement_limit=30, learning_rate=0.00035, betas=(0.5, .999), is_save=False, )
#
#     print("\nTraining CNN for ",list)
#
#     BATCH = 64
#     mnist_task_branch.create_and_train_CNN(model_id = name, num_epochs=15, batch_size=BATCH,  is_frozen=False,
#                                            is_off_shelf_model=True, epoch_improvement_limit=20, learning_rate=0.00025,
#                                            betas=(0.999, .999), is_save=True)
#     i +=1

#emnist_task_branch.create_and_train_VAE(model_id = label, num_epochs=EPOCHS, batch_size=BATCH, hidden_dim=10, latent_dim=ld, is_synthetic=False, is_take_existing_VAE=True, teacher_VAE=mnist_task_branch.VAE_most_recent, new_categories_to_add_to_existing_task= [], is_completely_new_task=True, epoch_improvement_limit=20, learning_rate=lr, betas=b, is_save=True)

do_proven_tests = False

if do_proven_tests:

    print("Generation, Classification")
    Utils.test_generating_and_classification_ability_multi_tasks(task_branch_list, number_per_category=10, device=device)


    print("New Task in new VAE: Pre-trained versus no pre-training")
    Utils.test_pre_trained_versus_non_pre_trained(emnist_task_branch, mnist_task_branch, model_id=label,
                                                  num_epochs=EPOCHS, batch_size=BATCH, hidden_dim=10, latent_dim=ld,
                                                  epoch_improvement_limit=20, learning_rate=lr, betas=b, is_save=False)
    print("Gate Allocation")
    gate.add_task_branch(mnist_task_branch,fashion_mnist_task_branch)
    Utils.test_gate_allocation(gate, mnist_data_and_interface,fashion_mnist_data_and_interface, number_tests_per_data_set=10000)

if is_grid_search:
    EPOCHS = 15
    latent_dim = [50, 75, 100]
    learning_rate = [0.00025, 0.0003, 0.00035]
    betas=[(0.1, .999),(0.3, .999),(0.5, .999)]

    for ld in latent_dim:
        for lr in learning_rate:
            for b in betas:
                print("\n***************************************\nnew hyperparameters  ")
                print("For MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b)
                mnist_task_branch.create_and_train_VAE(num_epochs=EPOCHS, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=20, learning_rate=lr, betas=b,is_save=False)
                print("For MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b)


                print("\n\nFor Fashion MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b)
                fashion_mnist_task_branch.create_and_train_VAE(num_epochs=EPOCHS, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=20, learning_rate=lr, betas=b, is_save=False)
                print("For Fashion MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b)

    learning_rate = [0.00025, 0.0003, 0.00035]
    betas=[(0.2, .999),(0.5, .999),(0.999, .999)]
    frozen_list = [True, False]

    for lr in learning_rate:
        for b in betas:
            for frozen in frozen_list:
                print("\n****************\nnew hyperparameters  ")
                print("For MNIST: frozen?", frozen," learning rate: ",lr, " betas ",b)
                mnist_task_branch.create_and_train_CNN(num_epochs=EPOCHS,  batch_size = 64,is_frozen=frozen, is_off_shelf_model = True, epoch_improvement_limit=20, learning_rate=lr, betas=b,is_save=False)
                print("For MNIST: frozen?", frozen, " learning rate: ", lr, " betas ", b)

                print("\nFor Fashion MNIST: frozen?", frozen, " learning rate: ", lr, " betas ", b)
                fashion_mnist_task_branch.create_and_train_CNN(num_epochs=EPOCHS,  batch_size = 64,is_frozen=frozen, is_off_shelf_model = True, epoch_improvement_limit=20, learning_rate=lr, betas=b,is_save=False)
                print("For Fashion MNIST: frozen?", frozen, " learning rate: ", lr, " betas ", b)

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

