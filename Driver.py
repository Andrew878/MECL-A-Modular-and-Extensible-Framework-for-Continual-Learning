import torch
import random
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import DatasetAndInterface as ds
import TaskBranch as task
import Utils
import Gate
import CustomDataSetAndLoaderForSynthetic
import RecordKeeper
from Invert import Invert
from copy import deepcopy

#%matplotlib inline

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
PATH_DATA_SVHN = str(PATH_ROOT) + "SVHN"
#PATH_MODELS = str(PATH_ROOT) + "properwithreg/"
PATH_MODELS = str(PATH_ROOT) + "properwithreg/"
PATH_MODELS_INITIAL = str(PATH_MODELS) + "saved_initial_models/"
record_keeper = RecordKeeper.RecordKeeper(PATH_MODELS)

dataset_path_list = [(datasets.MNIST,PATH_DATA_MNIST),(datasets.FashionMNIST,PATH_DATA_FashionMNIST),(datasets.EMNIST,PATH_DATA_EMNIST)]

# as suggested by pytorch devs
normalise_for_PIL_mean = [0.485, 0.456, 0.406]
normalise_for_PIL_std = [0.229, 0.224, 0.225]
normalise_for_PIL_mean = (0.5, 0.5, 0.5)
normalise_for_PIL_std = (0.5, 0.5, 0.5)
normalise_MNIST_mean = (0.1307,)
normalise_MNIST_std = (0.3081,)

image_height_MNIST = 28
image_channel_size_MNIST = 1

list_of_fixed_noise = []
for i in range(0,16):
    fixed_noise = torch.randn(1, 50).to(device)
    list_of_fixed_noise.append(fixed_noise)


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
        #transforms.RandomRotation(10),
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
    for x in ['train', 'val']:
        image_datasets_MNIST[x] = {}
        image_datasets_FashionMNIST[x] = {}
        for y in ['VAE','CNN']:
            image_datasets_MNIST[x][y] = datasets.MNIST(PATH_DATA_MNIST, train=(x == 'train'), download=True, transform = all_transforms[y][x])
            image_datasets_FashionMNIST[x][y] = datasets.FashionMNIST(PATH_DATA_FashionMNIST, train=(x == 'train'), download=True, transform = all_transforms[y][x])
            #print(image_datasets_MNIST[x][y].data)


def return_test_val_split(ds,num_labels,all_transform_dict, proportion =2, is_verbose = False, reduce_EMNIST = False):
    ds_length = len(ds)
    new_ds_val, new_ds_test = torch.utils.data.random_split(ds, [ds_length // proportion, ds_length - ds_length // proportion])
    new_ds_val = CustomDataSetAndLoaderForSynthetic.CustomDS(new_ds_val, all_transform_dict['val'], reduce_EMNIST)
    new_ds_test = CustomDataSetAndLoaderForSynthetic.CustomDS(new_ds_test, all_transform_dict['test'], reduce_EMNIST)
    print(len(new_ds_val),len(new_ds_test),)
    check_class_freq(new_ds_val,num_labels, is_verbose)
    check_class_freq(new_ds_test,num_labels, is_verbose)
    return new_ds_val, new_ds_test

def check_class_freq(ds,num_labels, is_verbose=False):
    freq_check = {i: 0 for i in range(0, num_labels)}
    average_per_label = len(ds)/num_labels
    for i in range(0, len(ds)):
        image, cat = ds[i]
        freq_check[cat] += 1
    if is_verbose:
        print("length dataset", len(ds))
        print(freq_check)
    for cat in freq_check:
        percent_variance = abs((freq_check[cat] - average_per_label) / average_per_label)
        if(percent_variance >0.20):
            print("Over ", percent_variance, "not stratified enough")

# is_verbose_checking_splits = True
# image_datasets_MNIST['test'] = {}
# image_datasets_FashionMNIST['test'] = {}
# for model in ['VAE','CNN']:
#     image_datasets_MNIST['val'][model],image_datasets_MNIST['test'][model] = return_test_val_split(image_datasets_MNIST['val'][model],num_labels=10, proportion=2,is_verbose=is_verbose_checking_splits, all_transform_dict=all_transforms[model])
#     image_datasets_FashionMNIST['val'][model],image_datasets_FashionMNIST['test'][model] = return_test_val_split(image_datasets_FashionMNIST['val'][model],num_labels=10, proportion=2,is_verbose=is_verbose_checking_splits,all_transform_dict=all_transforms[model])


mnist_data_and_interface = ds.DataSetAndInterface('MNIST', image_datasets_MNIST,PATH_DATA_MNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST, list_of_fixed_noise)
fashion_mnist_data_and_interface = ds.DataSetAndInterface('Fashion', image_datasets_FashionMNIST,PATH_DATA_FashionMNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST, list_of_fixed_noise)


transforms_CNN_one_channel_to_three_EMNIST = {
                'train': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'val': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
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

transforms_VAE_one_channel_EMNIST = {
                'train': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    transforms.Resize(image_height_MNIST),
                    #transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    # transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
                ]),
                'val': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                    # transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
                ]),
                'test': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                    # transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
                ]),
            }

transforms_CNN_SVHN = {
                'train': transforms.Compose([
                    transforms.Grayscale(),
                    Invert(),
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'val': transforms.Compose([
                    transforms.Grayscale(),
                    Invert(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test': transforms.Compose([
                    transforms.Grayscale(),
                    Invert(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
                'test_to_image': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(),
                    Invert(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(normalise_for_PIL_mean, normalise_for_PIL_std)
                ]),
            }

transforms_VAE_one_channel_SVHN = {
                'train': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    Invert(),
                    transforms.Resize(image_height_MNIST),
                    #transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    # transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
                ]),
                'val': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    Invert(),
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                    # transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
                ]),
                'test': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    Invert(),

                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                    # transforms.Normalize(normalise_MNIST_mean, normalise_MNIST_std)
                ]),
            }
all_transforms_EMNIST = {}
all_transforms_EMNIST['VAE'] = transforms_VAE_one_channel_EMNIST
all_transforms_EMNIST['CNN'] = transforms_CNN_one_channel_to_three_EMNIST
all_transforms_SVHN = {}
all_transforms_SVHN['VAE'] = transforms_VAE_one_channel_SVHN
all_transforms_SVHN['CNN'] = transforms_CNN_SVHN

image_datasets_EMNIST = {}
image_datasets_SVHN = {}
for x in ['train', 'val']:
    image_datasets_EMNIST[x] = {}
    image_datasets_SVHN[x] = {}

    for y in ['VAE', 'CNN']:
        image_datasets_EMNIST[x][y] = datasets.EMNIST(PATH_DATA_EMNIST, train=(x == 'train'), download=True,
                                                      transform=all_transforms_EMNIST[y][x], split='letters')
        # EMNIST has a index system starting at 1. All others have index system starting at 0. This makes an adjustment
        image_datasets_EMNIST[x][y].targets = image_datasets_EMNIST[x][y].targets - 1

        split = 'train'
        if(x=='val'):
            split = 'test'
        image_datasets_SVHN[x][y] = datasets.SVHN(PATH_DATA_SVHN,  download=True,
                                                      transform=all_transforms_SVHN[y][x], split=split)


# image_datasets_EMNIST['test'] ={}
# for model in ['VAE','CNN']:
#     print("EMNIST",model)
#     image_datasets_EMNIST['val'][model],image_datasets_EMNIST['test'][model] = return_test_val_split(image_datasets_EMNIST['val'][model],num_labels=26, proportion=2,is_verbose=is_verbose_checking_splits,all_transform_dict=all_transforms_EMNIST[model], reduce_EMNIST = True)


emnist_data_and_interface = ds.DataSetAndInterface('EMNIST', image_datasets_EMNIST,PATH_DATA_EMNIST,all_transforms_EMNIST, image_channel_size_MNIST, image_height_MNIST, list_of_fixed_noise)
svhn_data_and_interface = ds.DataSetAndInterface('SVHN', image_datasets_SVHN,PATH_DATA_SVHN,all_transforms_SVHN, image_channel_size_MNIST, image_height_MNIST, list_of_fixed_noise)
mnist_task_branch = task.TaskBranch('MNIST', mnist_data_and_interface, device,PATH_MODELS,record_keeper)
fashion_mnist_task_branch = task.TaskBranch('Fashion', fashion_mnist_data_and_interface, device, PATH_MODELS,record_keeper)
emnist_task_branch = task.TaskBranch('EMNIST', emnist_data_and_interface, device, PATH_MODELS,record_keeper)
svhn_task_branch = task.TaskBranch('SVHN', svhn_data_and_interface, device, PATH_MODELS,record_keeper)


is_saving = False
label = "v2"
BATCH = 128
EPOCHS = 100
ld = 50
b = (0.5, 0.999)
lr = 0.00035
wd = 0.0001
model_id = "final"

if is_saving:
    BATCH = 64


    EPOCHS_CNN = 50

    print("Training MNIST CNN")
    mnist_task_branch.create_and_train_CNN(model_id = model_id, num_epochs=EPOCHS_CNN, batch_size=BATCH,  is_frozen=False,
                                               is_off_shelf_model=True, epoch_improvement_limit=20, learning_rate=0.00025,
                                               betas=(0.999, .999),weight_decay=wd, is_save=True)
    print("Training Fashion MNIST CNN")
    fashion_mnist_task_branch.create_and_train_CNN(model_id = model_id, num_epochs=EPOCHS_CNN, batch_size=BATCH,  is_frozen=False,
                                                       is_off_shelf_model=True, epoch_improvement_limit=20, learning_rate=0.00025,
                                                       betas=(0.999, .999), weight_decay=wd, is_save=True)
    print("Training EMNIST CNN")

    emnist_task_branch.create_and_train_CNN(model_id = model_id, num_epochs=EPOCHS_CNN, batch_size=BATCH,  is_frozen=False,
                                           is_off_shelf_model=True, epoch_improvement_limit=20, learning_rate=0.00025,
                                           betas=(0.999, .999),weight_decay=wd, is_save=True)

    print("Training Fashion SVHN CNN")
    svhn_task_branch.create_and_train_CNN(model_id = model_id, num_epochs=EPOCHS_CNN, batch_size=BATCH,  is_frozen=False,
                                                   is_off_shelf_model=True, epoch_improvement_limit=20, learning_rate=0.00025,
                                                   betas=(0.999, .999), weight_decay=wd, is_save=True)

    EPOCHS_VAE = 100
    ld = 50
    b = (0.5,0.999)
    lr = 0.00035



    print("Training MNIST VAE")
    mnist_task_branch.create_and_train_VAE(model_id = model_id, num_epochs=EPOCHS_VAE, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=50, learning_rate=lr, betas=b,is_save=True, batch_size=BATCH)

    print("Training Fashion MNIST VAE")
    fashion_mnist_task_branch.create_and_train_VAE(model_id = model_id, num_epochs=EPOCHS_VAE, hidden_dim=10, latent_dim=ld,  is_synthetic=False,epoch_improvement_limit=50, learning_rate=lr, betas=b, is_save=True,batch_size=BATCH)

    print("Training EMNIST VAE")
    emnist_task_branch.create_and_train_VAE(model_id = model_id, num_epochs=EPOCHS_VAE, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=50, learning_rate=lr, betas=b,is_save=True, batch_size=BATCH)

    print("Training SVHN VAE")
    svhn_task_branch.create_and_train_VAE(model_id = model_id, num_epochs=EPOCHS_VAE, hidden_dim=10, latent_dim=ld,  is_synthetic=False,epoch_improvement_limit=50, learning_rate=lr, betas=b, is_save=True,batch_size=BATCH)

    mnist_task_branch.run_end_of_training_benchmarks("initial_training")
    fashion_mnist_task_branch.run_end_of_training_benchmarks("initial_training")
    emnist_task_branch.run_end_of_training_benchmarks("initial_training")
    svhn_task_branch.run_end_of_training_benchmarks("initial_training")
    record_keeper.record_to_file("initial_training")

else:
    print()
    is_update_mean_std = False
    mnist_task_branch.load_existing_VAE(PATH_MODELS_INITIAL+"VAE MNIST epochs100,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 100000000000 final",is_update_mean_std)
    fashion_mnist_task_branch.load_existing_VAE(PATH_MODELS_INITIAL+"VAE Fashion epochs100,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 100000000000 final", is_update_mean_std)

    mnist_task_branch.load_existing_CNN(
        PATH_MODELS_INITIAL + "CNN MNIST epochs50,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9968 final")
    fashion_mnist_task_branch.load_existing_CNN(
        PATH_MODELS_INITIAL + "CNN Fashion epochs50,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9419000000000001 final")

    emnist_task_branch.load_existing_VAE(PATH_MODELS_INITIAL+"VAE EMNIST epochs100,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 127.45233919192583 final",is_update_mean_std)
    svhn_task_branch.load_existing_VAE(PATH_MODELS_INITIAL+"VAE SVHN epochs100,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 493.7940254162998 final", is_update_mean_std)

    emnist_task_branch.load_existing_CNN(
        PATH_MODELS_INITIAL + "CNN EMNIST epochs50,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9559615384615384 final")
    svhn_task_branch.load_existing_CNN(
        PATH_MODELS_INITIAL + "CNN SVHN epochs50,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9726106330669944 final")




task_branch_list = [mnist_task_branch,fashion_mnist_task_branch,emnist_task_branch,svhn_task_branch]
gate = Gate.Gate()

for task in task_branch_list:
    task.generate_samples_to_display()



if False:
    Utils.test_gate_versus_non_gate(mnist_task_branch, fashion_mnist_task_branch, emnist_task_branch, svhn_task_branch)

#Utils.load_VAE_models_and_display_syn_images(PATH_MODELS,mnist_task_branch)

is_synthetic_tests = True
if (is_synthetic_tests):
    print("New and improved synthetic")
    #print("\n\n\n\n(x1.0 multiplier)....sigma is 0.5 FASHION")
    #Utils.test_synthetic_samples_versus_normal_increasing(fashion_mnist_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1)
    print("\n\n\n\n(x1.0 multiplier)....sigma is 0.5 SVHN")
    Utils.test_synthetic_samples_versus_normal_increasing(svhn_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1)
    print("\n\n\n\n(x1.0 multiplier)....sigma is 0.5 EMNIST")
    Utils.test_synthetic_samples_versus_normal_increasing(emnist_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1)
    #Utils.test_synthetic_samples_versus_normal_increasing_PRETRAINED_VAE(mnist_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1)
    #print("\n\n\n\nEXTRA SYNTHETIC SAMPLES (x0.8 multiplier)")
    #Utils.test_synthetic_samples_versus_normal_increasing(mnist_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1.25)

    #Utils.compare_pretrained_task_branches(mnist_data_and_interface, emnist_data_and_interface, PATH_MODELS,record_keeper)

    #
    # print("\n\n\n\nEXTRA SYNTHETIC SAMPLES (x0.8 multiplier)")
    # Utils.test_synthetic_samples_versus_normal(mnist_data_and_interface, emnist_data_and_interface, PATH_MODELS,record_keeper,device,extra_new_cat_multi=1.25)
    # print("\n\n\n\nEXTRA SYNTHETIC SAMPLES (x0.5 multiplier)")
    # Utils.test_synthetic_samples_versus_normal(mnist_data_and_interface, emnist_data_and_interface, PATH_MODELS,record_keeper,device,extra_new_cat_multi=2)
    # print("\n\n\n\nEXTRA SYNTHETIC SAMPLES (x1 multiplier)")
    # Utils.test_synthetic_samples_versus_normal(mnist_data_and_interface, emnist_data_and_interface, PATH_MODELS,record_keeper, device,extra_new_cat_multi=1)


print("\n\n\nCONCEPT DRIFT FOR FASHION AND SVHN\n\n\n")

if False:
    Utils.test_concept_drift_for_single_task(task_branch=svhn_task_branch, shear_degree_max=40,shear_degree_increments=10, split='train', num_samples_to_check=10000)
    Utils.test_concept_drift_for_single_task(task_branch=emnist_task_branch, shear_degree_max=40,shear_degree_increments=10, split='train', num_samples_to_check=10000)


if False:
    gate.add_task_branch(mnist_task_branch, fashion_mnist_task_branch, emnist_task_branch, svhn_task_branch)

    #print("Generation, Classification")
    #Utils.test_generating_and_classification_ability_multi_tasks(task_branch_list, number_per_category=1000, device=device)

    # print("Gate Allocation")
    # Utils.test_gate_allocation(gate, mnist_data_and_interface, fashion_mnist_data_and_interface, emnist_data_and_interface,
    #                            svhn_data_and_interface, number_tests_per_data_set=10000)

    print("Best fit for a dataset from Gate options")
    gate.given_new_dataset_find_best_fit_domain_from_existing_tasks(fashion_mnist_data_and_interface, [], 100)
    gate.given_new_dataset_find_best_fit_domain_from_existing_tasks(mnist_data_and_interface, [], 100)
    gate.given_new_dataset_find_best_fit_domain_from_existing_tasks(emnist_data_and_interface, [], 100)
    gate.given_new_dataset_find_best_fit_domain_from_existing_tasks(svhn_data_and_interface, [], 100)

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
    # print("Generation, Classification")
    # Utils.test_generating_and_classification_ability_multi_tasks(task_branch_list, number_per_category=10000, device=device)
    #
    # print("Gate Allocation")
    # gate.add_task_branch(mnist_task_branch,fashion_mnist_task_branch)
    # Utils.test_gate_allocation(gate, mnist_data_and_interface,fashion_mnist_data_and_interface,emnist_data_and_interface,number_tests_per_data_set=10000)

    for task_branch_to_be_created in [emnist_task_branch,svhn_task_branch]:
        print("Template is MNIST, learning ",task_branch_to_be_created.task_name)
        for sample_limit_multi in range(1,6):
            sample_limit = pow(10,sample_limit_multi)
            print("New Task in new VAE: Pre-trained versus no pre-training,sample_limit:", sample_limit)
            BATCH = min(sample_limit,50)
            Utils.test_pre_trained_versus_non_pre_trained(task_branch_to_be_created, mnist_task_branch, model_id=label,
                                                          num_epochs=EPOCHS, batch_size=BATCH, hidden_dim=10, latent_dim=ld,
                                                          epoch_improvement_limit=20, learning_rate=lr, betas=b, sample_limit=sample_limit,is_save=False)

    for task_branch_to_be_created in [emnist_task_branch,svhn_task_branch]:
        print("Template is Fashion MNIST, learning ", task_branch_to_be_created.task_name)
        for sample_limit_multi in range(1,6):
                sample_limit = pow(10,sample_limit_multi)
                print("New Task in new VAE: Pre-trained versus no pre-training,sample_limit:", sample_limit)
                BATCH = min(sample_limit,50)
                Utils.test_pre_trained_versus_non_pre_trained(task_branch_to_be_created, fashion_mnist_task_branch, model_id=label,
                                                          num_epochs=EPOCHS, batch_size=BATCH, hidden_dim=10, latent_dim=ld,
                                                          epoch_improvement_limit=20, learning_rate=lr, betas=b, sample_limit=sample_limit,is_save=False)
is_grid_search = False

if is_grid_search:
    EPOCHS = 15
    latent_dim = [50, 75]
    weigh_decay = [0,0.0001,0.01,0.1]
    lr = 0.00035
    betas=[(0.3, .999),(0.5, .999)]

    for ld in latent_dim:
        for b in betas:
            for wd in weigh_decay:

                print("\n***************************************\nnew hyperparameters  ")
                print("For MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b, "weight_decay", wd)
                mnist_task_branch.create_and_train_VAE(model_id="grid_search", num_epochs=EPOCHS, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=20, learning_rate=lr, betas=b,weight_decay=wd,is_save=False)

                print("\n\nFor Fashion MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b, "weight_decay", wd)
                fashion_mnist_task_branch.create_and_train_VAE(model_id="grid_search",num_epochs=EPOCHS, hidden_dim=10, latent_dim=ld, is_synthetic=False,epoch_improvement_limit=20, learning_rate=lr, betas=b,weight_decay=wd, is_save=False)
#                print("For Fashion MNIST: latent dimensions", ld," learning rate: ",lr, " betas ",b, "weight_decay", wd)

    #learning_rate = [0.00025, 0.0003, 0.00035]
    betas=[(0.999, .999)]
    wd = [0, 0.0001,0.01,0.1]
    #frozen = False#[True, False]
    frozen_list = [True, False]
    EPOCHS = 5

    for wd in weigh_decay:
        for b in betas:
            for frozen in frozen_list:
                #for wd in weigh_decay:

                print("\n****************\nnew hyperparameters  ")
                print("For MNIST: frozen?", frozen," learning rate: ",lr, " betas ",b, "weight_decay", wd)
                mnist_task_branch.create_and_train_CNN(model_id="grid_search",num_epochs=EPOCHS,  batch_size = 64,is_frozen=frozen, is_off_shelf_model = True, epoch_improvement_limit=20, learning_rate=lr, betas=b,weight_decay=wd,is_save=False)
         #       print("For MNIST: frozen?", frozen," learning rate: ",lr, " betas ",b, "weight_decay", wd)

                print("\nFor Fashion MNIST: frozen?", frozen," learning rate: ",lr, " betas ",b, "weight_decay", wd)
                fashion_mnist_task_branch.create_and_train_CNN(model_id="grid_search",num_epochs=EPOCHS,  batch_size = 64,is_frozen=frozen, is_off_shelf_model = True, epoch_improvement_limit=20, learning_rate=lr, betas=b,weight_decay=wd,is_save=False)
          #      print("For Fashion MNIST: frozen?", frozen," learning rate: ",lr, " betas ",b, "weight_decay", wd)

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

