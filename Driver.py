import torch
import random
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF

import DatasetAndInterface as ds
import TaskBranch as task
import Utils
import Gate
import RecordKeeper
from Invert import Invert

# uncomment to plot in Jupyter
#%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manualSeed = random.randint(1, 10000)

################################ USER INPUT REQUIRED ####################################################

# fixed random seed
manualSeed = 149
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# enter folder paths
PATH_ROOT = "/cs/tmp/al278/"
PATH_ROOT = "/cs/scratch/al278/"

# True if running code for first time
is_train_and_save_new_models = False

#
# TEST THE FOLLOWING FUNCTIONALITY
is_gen_classify_test = False
is_gate_tests = False
is_continual_learning_incremental_categories_test = False
is_concept_drift_test = True
is_test_transfer_learning_for_new_domain = False

# to speed up testing but still demonstrate functionality
is_fast_testing = True

################################ USER INPUT NO  LONGER REQUIRED #########################################


PATH_DATA_MNIST = str(PATH_ROOT)+"MNIST"
PATH_DATA_FashionMNIST = str(PATH_ROOT) + "FashionMNIST"
PATH_DATA_EMNIST = str(PATH_ROOT) + "EMNIST/EMNIST"
PATH_DATA_SVHN = str(PATH_ROOT) + "SVHN"
PATH_MODELS = str(PATH_ROOT) + "properwithreg1/"
PATH_MODELS_INITIAL = str(PATH_MODELS) + "saved_initial_models/"
dataset_path_list = [(datasets.MNIST,PATH_DATA_MNIST),(datasets.FashionMNIST,PATH_DATA_FashionMNIST),(datasets.EMNIST,PATH_DATA_EMNIST)]

# create record keeper object
record_keeper = RecordKeeper.RecordKeeper(PATH_MODELS)


# initalise variables

# as suggested by pytorch devs
normalise_for_PIL_mean = (0.5, 0.5, 0.5)
normalise_for_PIL_std = (0.5, 0.5, 0.5)
normalise_MNIST_mean = (0.1307,)
normalise_MNIST_std = (0.3081,)
image_height_MNIST = 28
image_channel_size_MNIST = 1

# create list of fixed noise so we can observe synthetics samples evolving
list_of_fixed_noise = []
for i in range(0,16):
    fixed_noise = torch.randn(1, 50).to(device)
    list_of_fixed_noise.append(fixed_noise)


# transforms. Note 'val' acts as a test set, not validation set
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
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_height_MNIST ),
        transforms.ToTensor(),
    ]),
}


# add transforms to dictionary for easy access
all_transforms = {}
all_transforms['VAE'] = transforms_VAE_one_channel
all_transforms['CNN'] = transforms_CNN_one_channel_to_three

# create dictionaries of datasets. Seperate datasets for CNN and VAE are required because of differences in dimensionality
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


# create datasetAndInterface objects for MNIST and Fashion (they both have the same dimensions, transformations)
mnist_data_and_interface = ds.DataSetAndInterface('MNIST', image_datasets_MNIST,PATH_DATA_MNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST, list_of_fixed_noise)
fashion_mnist_data_and_interface = ds.DataSetAndInterface('Fashion', image_datasets_FashionMNIST,PATH_DATA_FashionMNIST,all_transforms, image_channel_size_MNIST, image_height_MNIST, list_of_fixed_noise)

# EMNIST requries its own special transform, because images are rotated and flipped
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
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
            }


# SVHN requires its own unique transformations due to dimensionality
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
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    Invert(),
                    transforms.Resize(image_height_MNIST),
                    transforms.ToTensor(),
                ]),
            }

# store transformations in dictionaries
all_transforms_EMNIST = {}
all_transforms_EMNIST['VAE'] = transforms_VAE_one_channel_EMNIST
all_transforms_EMNIST['CNN'] = transforms_CNN_one_channel_to_three_EMNIST
all_transforms_SVHN = {}
all_transforms_SVHN['VAE'] = transforms_VAE_one_channel_SVHN
all_transforms_SVHN['CNN'] = transforms_CNN_SVHN

# store datasets in dictionaries
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


# create datasetAndInterface objects for EMNIST and SVHN
emnist_data_and_interface = ds.DataSetAndInterface('EMNIST', image_datasets_EMNIST,PATH_DATA_EMNIST,all_transforms_EMNIST, image_channel_size_MNIST, image_height_MNIST, list_of_fixed_noise)
svhn_data_and_interface = ds.DataSetAndInterface('SVHN', image_datasets_SVHN,PATH_DATA_SVHN,all_transforms_SVHN, image_channel_size_MNIST, image_height_MNIST, list_of_fixed_noise)

# create task branch objects for our Domains
mnist_task_branch = task.TaskBranch('MNIST', mnist_data_and_interface, device,PATH_MODELS,record_keeper)
fashion_mnist_task_branch = task.TaskBranch('Fashion', fashion_mnist_data_and_interface, device, PATH_MODELS,record_keeper)
emnist_task_branch = task.TaskBranch('EMNIST', emnist_data_and_interface, device, PATH_MODELS,record_keeper)
svhn_task_branch = task.TaskBranch('SVHN', svhn_data_and_interface, device, PATH_MODELS,record_keeper)


# train our initial task branches using following parameters
label = "v2"
BATCH = 128
EPOCHS = 100
ld = 50
b = (0.5, 0.999)
lr = 0.00035
wd = 0.0001
model_id = "final"

if is_train_and_save_new_models:
    BATCH = 64

    EPOCHS_CNN = 50
    EPOCHS_VAE = 100

    if is_fast_testing:
        EPOCHS_CNN = 2
        EPOCHS_VAE = 5

    # TRAIN AND SAVE CNN'S

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


    # TRAIN AND SAVE VAE'S

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

    # RUN ACCURACY BENCHMARKS ON INDIVIDUAL BRANCHES
    mnist_task_branch.run_end_of_training_benchmarks("initial_training")
    fashion_mnist_task_branch.run_end_of_training_benchmarks("initial_training")
    emnist_task_branch.run_end_of_training_benchmarks("initial_training")
    svhn_task_branch.run_end_of_training_benchmarks("initial_training")


# IF NOT TRAINING NEW MODELS, THEN LOAD PREVIOUSLY SAVED VERSIONS
else:
    # a flag to avoid recalculating expensive mean/standard deviation category information (if false, then we load previously saved mean/standard deviation cat info)
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





# create list of tasks
task_branch_list = [mnist_task_branch,fashion_mnist_task_branch,emnist_task_branch,svhn_task_branch]
gate = Gate.Gate()
gate.add_task_branch(mnist_task_branch, fashion_mnist_task_branch, emnist_task_branch, svhn_task_branch)


###
# RUN TESTS
###

if is_gen_classify_test:
    print("Generation, Classification")

    if is_fast_testing:
        number_per_category = 10
    else:
        number_per_category = 1000

    Utils.test_generating_and_classification_ability_multi_tasks(task_branch_list, number_per_category=number_per_category, device=device)



if is_gate_tests:
    print("\nGate Allocation WITH RANDOM")
    Utils.test_gate_allocation(gate, mnist_data_and_interface, fashion_mnist_data_and_interface, emnist_data_and_interface,svhn_data_and_interface, number_tests_per_data_set=100)

    print("\nBest fit for a dataset from Gate options. Entire test set")
    mnist_size = 10
    fashion_size = 10
    emnist_size = 10
    svhn_size = 10

    if not is_fast_testing:
        mnist_size = mnist_data_and_interface.val_set_size
        fashion_size = fashion_mnist_data_and_interface.val_set_size
        emnist_size = emnist_data_and_interface.val_set_size
        svhn_size = svhn_data_and_interface.val_set_size

    gate.given_new_dataset_find_best_fit_domain_from_existing_tasks(mnist_data_and_interface, [], mnist_size)
    gate.given_new_dataset_find_best_fit_domain_from_existing_tasks(fashion_mnist_data_and_interface, [], fashion_size)
    gate.given_new_dataset_find_best_fit_domain_from_existing_tasks(emnist_data_and_interface, [], emnist_size)
    gate.given_new_dataset_find_best_fit_domain_from_existing_tasks(svhn_data_and_interface, [], svhn_size)

    print("\nGate versus non-gate (tests on entire test set....can take a while)")
    Utils.test_gate_versus_non_gate(mnist_task_branch, fashion_mnist_task_branch, emnist_task_branch, svhn_task_branch)


if is_continual_learning_incremental_categories_test:
    print("\nCONTINUAL LEARNING TEST - INCREMENTAL CATEGORIES WITH PSUEDO-REHEARSAL")

    # only test one if fast testing chosen
    print("\n\n\n\n(x1.0 multiplier)....sigma is 0.5 MNIST")
    Utils.test_synthetic_samples_versus_normal_increasing(mnist_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1, is_fast_testing=is_fast_testing)

    if not is_fast_testing:
        print("\n\n\n\n(x1.0 multiplier)....sigma is 0.5 Fashion MNIST")
        Utils.test_synthetic_samples_versus_normal_increasing(fashion_mnist_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1, is_fast_testing=is_fast_testing)
        print("\n\n\n\n(x1.0 multiplier)....sigma is 0.5 EMNIST")
        Utils.test_synthetic_samples_versus_normal_increasing(emnist_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1, is_fast_testing=is_fast_testing)
        print("\n\n\n\n(x1.0 multiplier)....sigma is 0.5 SVHN")
        Utils.test_synthetic_samples_versus_normal_increasing(svhn_data_and_interface,PATH_MODELS,record_keeper,extra_new_cat_multi=1, is_fast_testing=is_fast_testing)

if is_concept_drift_test:
    print("\n\n\nCONCEPT DRIFT TEST\n\n\n")
    num_samples_to_check = 10000

    for task in task_branch_list:
        Utils.test_concept_drift_for_single_task(task_branch=task, shear_degree_max=30,shear_degree_increments=10, split='train', num_samples_to_check=num_samples_to_check)


if is_test_transfer_learning_for_new_domain:

    # to speed up demonstration make epochs 3,
    EPOCHS = 50
    if is_fast_testing:
        EPOCHS = 3

    # use MNIST as template
    for task_branch_to_be_created in [emnist_task_branch,svhn_task_branch]:
        print("Template is MNIST, learning ",task_branch_to_be_created.task_name)
        for sample_limit_multi in range(1,6):
            sample_limit = pow(10,sample_limit_multi)
            sum_pre_train = 0
            sum_scratch = 0
            if sample_limit <= 1000:
                num_times_to_take_test = 10
            else:
                num_times_to_take_test = 1

            for i in range(0,num_times_to_take_test):
                print("New Task in new VAE: Pre-trained versus no pre-training,sample_limit:", sample_limit, "Number ", i)
                BATCH = min(sample_limit,50)
                average_train_loss_pre_train, average_train_loss_scratch = Utils.test_pre_trained_versus_non_pre_trained(task_branch_to_be_created, mnist_task_branch, model_id=label,
                                                                                                                         num_epochs=EPOCHS, batch_size=BATCH, hidden_dim=10, latent_dim=ld,
                                                                                                                         epoch_improvement_limit=20, learning_rate=lr, betas=b, sample_limit=sample_limit,is_save=False)
                sum_pre_train += average_train_loss_pre_train
                sum_scratch += average_train_loss_scratch
            print("\n\n_______\nsample_limit:", sample_limit)
            print("pretrain average ", sum_pre_train/10)
            print("scratch average ", sum_scratch/10)


    # use Fashion as template
    for task_branch_to_be_created in [emnist_task_branch,svhn_task_branch]:
        print("Template is Fashion MNIST, learning ", task_branch_to_be_created.task_name)
        for sample_limit_multi in range(1,6):
            sample_limit = pow(10,sample_limit_multi)
            sum_pre_train = 0
            sum_scratch = 0
            if sample_limit <= 1000:
                num_times_to_take_test = 10
            else:
                num_times_to_take_test = 1

            for i in range(0,num_times_to_take_test):
                print("New Task in new VAE: Pre-trained versus no pre-training,sample_limit:", sample_limit,"Number ", i)
                BATCH = min(sample_limit, 50)
                average_train_loss_pre_train, average_train_loss_scratch = Utils.test_pre_trained_versus_non_pre_trained(task_branch_to_be_created, fashion_mnist_task_branch, model_id=label,
                                                                                                                         num_epochs=EPOCHS, batch_size=BATCH, hidden_dim=10, latent_dim=ld,
                                                                                                                         epoch_improvement_limit=20, learning_rate=lr, betas=b, sample_limit=sample_limit,is_save=False)
                sum_pre_train += average_train_loss_pre_train
                sum_scratch += average_train_loss_scratch
            print("\n\n_______\nsample_limit:", sample_limit)
            print("pretrain average ", sum_pre_train / 10)
            print("scratch average ", sum_scratch / 10)


# below shows some of the code used for grid search. Can be ignored

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
    #frozen = [True, False]
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



# the below is code for running test some  pre-trained models. Can be ignored

is_testing_presaved_models = False

if is_testing_presaved_models:
    model_string_vae = []
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 101.1282752212213 increment0synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 78.45619751514317 increment1synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 89.60624052141372 increment2synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 95.91765151826462 increment3synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 98.84272441048243 increment4synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 102.96877034505208 increment5synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 104.12568975369538 increment6synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 104.48726003921449 increment7synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 108.8118243938762 increment8synth multi 1")
    model_string_vae.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 109.9276000371655 increment9synth multi 1")

    model_string_cnn = []
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment0synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment1synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment2synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment3synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment4synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment5synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment6synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment7synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9998859228838695 increment8synth multi 1")
    model_string_cnn.append("CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment9synth multi 1")

    Utils.test_synthetic_samples_versus_normal_increasing_PRETRAINED_VAE(mnist_data_and_interface,PATH_MODELS,model_string_vae, model_string_cnn,record_keeper,extra_new_cat_multi=1)

    model_string_vae = []
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 268.649826171875 increment0synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 222.38356726074218 increment1synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 248.73671929253473 increment2synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 242.9933427734375 increment3synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 252.5843380859375 increment4synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 245.6725674641927 increment5synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 256.5289255894252 increment6synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 250.1580067952474 increment7synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 258.0115279224537 increment8synth multi 1")
    model_string_vae.append("VAE Fashion pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 258.78205587565105 increment9synth multi 1")

    model_string_cnn = []
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment0synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment1synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9998888888888889 increment2synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9996666666666666 increment3synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9994666666666667 increment4synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9998055555555556 increment5synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9997142857142858 increment6synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9997291666666667 increment7synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9997407407407407 increment8synth multi 1")
    model_string_cnn.append("CNN Fashion pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9996 increment9synth multi 1")


    Utils.test_synthetic_samples_versus_normal_increasing_PRETRAINED_VAE(fashion_mnist_data_and_interface,PATH_MODELS,model_string_vae, model_string_cnn,record_keeper,extra_new_cat_multi=1)

    model_string_vae = []
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 150.01349283854168 increment0synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 162.12529490152994 increment1synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 167.70221320258247 increment2synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 169.85602737426757 increment3synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 176.59117801920573 increment4synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 175.96653700086804 increment5synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 179.36047659737724 increment6synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 180.07620900472006 increment7synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 176.69265882703993 increment8synth multi 1")
    model_string_vae.append("VAE EMNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 173.73905775960287 increment9synth multi 1")

    model_string_cnn = []
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment0synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment1synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9994444444444445 increment2synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9996354166666668 increment3synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9993333333333333 increment4synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9993055555555556 increment5synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9994940476190476 increment6synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.99875 increment7synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9992361111111111 increment8synth multi 1")
    model_string_cnn.append("CNN EMNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.999375 increment9synth multi 1")

    Utils.test_synthetic_samples_versus_normal_increasing_PRETRAINED_VAE(emnist_data_and_interface,PATH_MODELS,model_string_vae, model_string_cnn,record_keeper,extra_new_cat_multi=1)

    model_string_vae = []
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 499.4539507755659 increment0synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 500.3527360518045 increment1synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 501.9317554721943 increment2synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 504.4524426374405 increment3synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 505.3855414278292 increment4synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 506.82711927684994 increment5synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 507.61129331475746 increment6synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 508.0052609141114 increment7synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 509.8751990333526 increment8synth multi 1")
    model_string_vae.append("VAE SVHN pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 509.15853640132 increment9synth multi 1")

    model_string_cnn = []
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment0synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9999999999999999 increment1synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9998740355849473 increment2synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9994704013181123 increment3synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9994368463395013 increment4synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9994914269107817 increment5synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9994262765347103 increment6synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9993074173369079 increment7synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9994714238519987 increment8synth multi 1")
    model_string_cnn.append("CNN SVHN pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.999570723331187 increment9synth multi 1")

    Utils.test_synthetic_samples_versus_normal_increasing_PRETRAINED_VAE(svhn_data_and_interface,PATH_MODELS,model_string_vae, model_string_cnn,record_keeper,extra_new_cat_multi=1)
