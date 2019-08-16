import torch
import TaskBranch as task
import copy
import Gate
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import random

"""This file contains a various ubiquitous utility functions and also many of the tests/evaluation procedures."""

def idx2onehot(idx, num_categories):
    """Returns a one_hot_encoded_vector. Adapted from https://graviraja.github.io/conditionalvae/#"""

    # Check dimensionality
    assert idx.shape[1] == 1
    assert torch.max(idx).item() < num_categories
    onehot = torch.zeros(idx.size(0), num_categories)
    onehot.scatter_(1, idx.data, 1)
    return onehot


def generate_list_of_class_category_tensors(N_CLASSES):
    """Generates a list of category tensors """

    class_list = []
    for j in range(0, N_CLASSES):
        class_list.append(torch.tensor(np.array([[j], ])).to(dtype=torch.long))
    return class_list


def test_generating_and_classification_ability_multi_tasks(task_branches, number_per_category=1,device= 'cuda'):

    """Takes multiple task branches, performs the generation/classification test"""

    print("\n************ Testing generation and classification matches:")

    # cycle through each task branch
    for task_branch in task_branches:

        # obtain correct matches by task branch
        correct_matches_by_class = test_generating_and_classification_ability_single_task(number_per_category,
                                                                                          task_branch, device)
        # print results
        test_generator_accuracy(correct_matches_by_class, number_per_category)


def test_generating_and_classification_ability_single_task(number_per_category, task_branch, device='cuda'):

    """For a single task, generates a specified number of synthetic samples (per category) and then checks if the classifier labels them correctly"""

    print("\nTask: ", task_branch.task_name)
    correct_matches_by_class = {i: 0 for i in range(0, task_branch.num_categories_in_task)}

    # generate the samples
    synthetic_data_list_x, synthetic_data_list_y, = task_branch.VAE_most_recent.generate_synthetic_set_all_cats(
        number_per_category=number_per_category)

    # classify the images and return the matches
    for i in range(0, len(synthetic_data_list_x)):

        img_transformed = task_branch.dataset_interface.transformations['CNN']['test_to_image'](
            synthetic_data_list_x[i].cpu()).float()

        # None is because model is expecting a batch, and we want only a single observation. Also need to send to CUDA
        pred, _ = task_branch.classify_with_CNN(img_transformed[None].to(device))
        if (pred.item() == synthetic_data_list_y[i]):
            correct_matches_by_class[synthetic_data_list_y[i]] += 1

    return correct_matches_by_class


def test_generator_accuracy(accuracy_score_by_class, number_per_category):

    """A helper function that prints the print generator accuracy"""

    total = 0
    for category in accuracy_score_by_class:
        total += accuracy_score_by_class[category]
        accuracy = 1.0 * accuracy_score_by_class[category] / (1.0 * number_per_category)
        print('Class: {}, % of VAE-CNN matches: {:.00%} out of {}'.format(category, accuracy, number_per_category))
    total_accuracy = total / (len(accuracy_score_by_class) * number_per_category)
    print('All classes overall, % of VAE-CNN matches: {:%} out of {}'.format(total_accuracy, (
        len(accuracy_score_by_class)) * number_per_category))


def test_gate_allocation(gate, *data_set_interfaces, number_tests_per_data_set=1000):

    """Tests how well the Gate allocates samples to the correct dataset.
    Performs this using both absolute and relative measures"""

    # initialise overall variables
    overall_correct_task_allocations_recon = 0
    overall_correct_task_allocations_std = 0
    overall_correct_task_and_class_allocations_recon = 0
    overall_correct_task_and_class_allocations_std = 0
    num_tests_overall = 0

    # cycle through each dataset
    for data_set_interface in data_set_interfaces:

        print("\n ********  For samples taken from ", data_set_interface.name)

        data_loaders = data_set_interface.return_data_loaders(branch_component='VAE', BATCH_SIZE=1)
        num_tests = 0

        # initialise per dataset variables
        correct_task_allocation_recon = 0
        correct_task_and_class_allocation_recon = 0
        correct_task_allocation_std = 0
        correct_task_and_class_allocation_std = 0

        # pass samples through the gate and allocate to the appropriate task, then make a record
        for i, (x, y) in enumerate(data_loaders['val']):

            best_task_recon, best_class_recon, best_task_std, best_class_std,__ = gate.allocate_sample_to_task_branch(x,
                                                                                                                   is_standardised_distance_check=True,
                                                                                                                   is_return_both_metrics=True)
            # Absolute measure: if match between task and/or category, make a record
            if (best_task_recon.task_name == data_set_interface.name):
                correct_task_allocation_recon += 1
                if (best_class_recon == y.item()):
                    correct_task_and_class_allocation_recon += 1

            # relative measure: if match between task and/or category, make a record
            if (best_task_std.task_name == data_set_interface.name):
                correct_task_allocation_std += 1
                if (best_class_std == y.item()):
                    correct_task_and_class_allocation_std += 1

            num_tests += 1
            if number_tests_per_data_set <= num_tests:
                break

        overall_correct_task_allocations_recon += correct_task_allocation_recon
        overall_correct_task_allocations_std += correct_task_allocation_std

        overall_correct_task_and_class_allocations_recon += correct_task_and_class_allocation_recon
        overall_correct_task_and_class_allocations_std += correct_task_and_class_allocation_std
        num_tests_overall += num_tests

        # print results per dataset
        gate_accuracy_print_results(correct_task_allocation_recon, correct_task_allocation_std,
                                    correct_task_and_class_allocation_recon, correct_task_and_class_allocation_std,
                                    num_tests)

    # print overall results
    print("\n ********  Overall Results")
    gate_accuracy_print_results(overall_correct_task_allocations_recon, overall_correct_task_allocations_std,
                                overall_correct_task_and_class_allocations_recon,
                                overall_correct_task_and_class_allocations_std,
                                num_tests_overall)


def gate_accuracy_print_results(correct_task_allocation_recon, correct_task_allocation_std,
                                correct_task_and_class_allocation_recon, correct_task_and_class_allocation_std,
                                num_tests):

    """Prints accuracy of the gate allocation"""

    print("Out of", num_tests, "samples.")
    print(
        'Reconstruction error:   {} correct, % of correct allocations: {:.00%}, % of correct allocations and class: {:.00%}'.format(
            correct_task_allocation_recon, correct_task_allocation_recon / num_tests,
                                           correct_task_and_class_allocation_recon / num_tests))
    print(
        'Std dev distance error: {} correct, % of correct allocations: {:.00%}, % of correct allocations and class: {:.00%}'.format(
            correct_task_allocation_std, correct_task_allocation_std / num_tests,
                                         correct_task_and_class_allocation_std / num_tests))

def test_gate_versus_non_gate(*task_branches, number_tests_per_data_set=1000):

    """Tests how well the Agent with Gate classifies relative to a 'most confident' model"""

    print("*********** Testing Gate versus non-gate_method")

    # initialise variables
    gate = Gate.Gate()
    dataset_list = []
    confusion_matrix =  {}
    for task in task_branches:
        gate.add_task_branch(task)
        dataset_list.append(task.dataset_interface)
        confusion_matrix[task.task_name] = {}
    GATE_overall_correct_task_classification = 0
    NO_GATE_overall_correct_task_classification = 0
    num_tests_overall = 0


    # test accuracy on each dataset
    for data_set_interface in dataset_list:
        per_task_GATE_overall_correct_task_classification = 0
        per_task_NO_GATE_overall_correct_task_classification = 0
        per_task_num_tests = 0

        # initialise row of confusion matrix
        confusion_matrix[data_set_interface.name] = {task.task_name:0 for task in task_branches}

        dataloader = data_set_interface.return_data_loaders('VAE', BATCH_SIZE=1)

        # test accuracy for both methods
        for i, (x, y) in enumerate(dataloader['val']):

            num_tests_overall += 1
            per_task_num_tests += 1

            # TEST ONE: classify with gate
            best_task_recon, pred_cat, __ = gate.classify_input_using_allocation_method(x)

            # update confusion matrix
            confusion_matrix[data_set_interface.name][best_task_recon.task_name] += 1

            # if a match, update accuracy
            if (best_task_recon.task_name == data_set_interface.name):
                if (pred_cat == y.item()):
                    GATE_overall_correct_task_classification += 1
                    per_task_GATE_overall_correct_task_classification +=1

            highest_prob = float('-inf')
            highest_prob_cat = 0
            highest_prob_task = None
            x_original = x

            # TEST TWO: classify with most confident
            # cycle through each task and choose most probable class
            for task in task_branches:
                x = task.dataset_interface.transformations['CNN']['test_to_image'](torch.squeeze(copy.deepcopy(x_original)))
                x = torch.unsqueeze(x, 0)
                pred_cat, probability = task.classify_with_CNN(x)
                if probability>highest_prob:
                    highest_prob = probability
                    highest_prob_cat = pred_cat.item()
                    highest_prob_task = task

            if (highest_prob_task.task_name == data_set_interface.name):
                if (highest_prob_cat == y.item()):
                    NO_GATE_overall_correct_task_classification += 1
                    per_task_NO_GATE_overall_correct_task_classification += 1

        # print per dataset results
        print(data_set_interface.name," by task samples: ",per_task_num_tests, " Accuracy with Gate: ",per_task_GATE_overall_correct_task_classification/per_task_num_tests, " Accuracy without gate: ",per_task_NO_GATE_overall_correct_task_classification/per_task_num_tests)

    # print overall results
    print("Overall Samples: ", num_tests_overall, " Gate accuracy: ",GATE_overall_correct_task_classification / num_tests_overall, " Accuracy without gate: ",NO_GATE_overall_correct_task_classification / num_tests_overall,"\n\n")

    print("*** Printing confusion matrix ***")
    print("For dataset", confusion_matrix[dataset_list[0].name].keys())
    for task in task_branches:
        print("Best task is",task.task_name, confusion_matrix[task.task_name].values())


def test_pre_trained_versus_non_pre_trained(new_task_to_be_trained, template_task, model_id, num_epochs=50,
                                            batch_size=64, hidden_dim=10, latent_dim=50, epoch_improvement_limit=20,
                                            learning_rate=0.00035, betas=(0.5, .999), sample_limit = float('Inf'), is_save=False):

    """Tests how well transfer learning performs on a new Domain. Trains two VAE's, one with pre-training another with random initialisation"""

    # make copies
    new_task_to_be_trained = copy.deepcopy(new_task_to_be_trained)
    template_task = copy.deepcopy(template_task)

    # PERFORM TRAINING WITH PRETRAINED MODEL
    print("--- Task to be trained: ", new_task_to_be_trained.task_name)
    print("*********** Training with pretrained weights from: ", template_task.task_name)

    new_task_to_be_trained.create_and_train_VAE(model_id=model_id + "untrained", num_epochs=num_epochs,
                                                batch_size=batch_size,
                                                hidden_dim=10,
                                                latent_dim=latent_dim, is_take_existing_VAE=True,
                                                teacher_VAE=template_task.VAE_most_recent, is_completely_new_task=True,
                                                epoch_improvement_limit=epoch_improvement_limit,
                                                learning_rate=learning_rate, betas=betas, sample_limit = sample_limit,is_save=is_save)

    average_train_loss_pre_train =  new_task_to_be_trained.overall_average_reconstruction_error

    # PERFORM TRAINING WITH RANDOM MODEL
    print("*********** Training from scratch ")

    new_task_to_be_trained.create_and_train_VAE(model_id=model_id + "pretrained", num_epochs=num_epochs,
                                                batch_size=batch_size, hidden_dim=10,
                                                latent_dim=latent_dim, is_take_existing_VAE=False,
                                                teacher_VAE=None, is_completely_new_task=True,
                                                epoch_improvement_limit=epoch_improvement_limit,
                                                learning_rate=learning_rate, betas=betas,sample_limit = sample_limit, is_save=is_save)

    average_train_loss_scratch =  new_task_to_be_trained.overall_average_reconstruction_error

    # return metrics
    return average_train_loss_pre_train, average_train_loss_scratch



def test_synthetic_samples_versus_normal_increasing(original_task_datasetInterface, PATH_MODELS,record_keeper,
                                         device='cuda',number_increments=10, extra_new_cat_multi=1, is_fast_testing = False):

    """Tests pseudo rehearsal methods relative to training with real samples. Domains start with a single category Domain, and then it increasing one by one
     Pseudo-rehearsal methods replay past categories from VAE samples, the real sample method uses original data"""

    print("***** Testing pseudo-rehearsal versus real samples")

    # initialise variables
    is_save = True
    BATCH = 64
    EPOCH_IMPROVE_LIMIT = 20
    BETA_CNN = (0.999, .999)
    LEARNR_CNN = 0.00025
    BETA_VAE = (0.5, 0.999)
    LAT_DIM_VAE = 50
    LEARNR_VAE = 0.00035
    combined_task_branch_synthetic = None
    cat_list_all = original_task_datasetInterface.categories_list
    EPOCH_CNN = 10
    EPOCH_VAE = 50

    # to shorten training in order to demonstrate functionality
    if is_fast_testing:
        EPOCH_CNN = 2
        EPOCH_VAE = 2



    # Increase categories one by one and train data sets.
    for increment in range(0, number_increments):

        task_DIS_orig = copy.deepcopy(original_task_datasetInterface)
        category_list_subset = cat_list_all[0:increment+1]
        task_DIS_orig.reduce_categories_in_dataset(category_list_subset)
        task_DIS_orig_for_synth = copy.deepcopy(task_DIS_orig)

        name = "increment" + str(increment) + "synth multi "+ str(extra_new_cat_multi)

        # For the first category, only real samples are used
        if (increment == 0):
            combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig_for_synth.name + " pseudo", task_DIS_orig_for_synth,
                                                             device, PATH_MODELS,record_keeper)

            print("Starting point, just ....")
            print("Training VAE - Starting")

            print("\n Beginning point only ",category_list_subset)

            combined_task_branch_synthetic.create_and_train_VAE(model_id=name, num_epochs=EPOCH_VAE, batch_size=BATCH,
                                                                hidden_dim=10,
                                                                latent_dim=LAT_DIM_VAE,
                                                                is_synthetic=False, is_take_existing_VAE=False,
                                                                teacher_VAE=None,
                                                                is_new_categories_to_addded_to_existing_task=False,
                                                                is_completely_new_task=True,
                                                                epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                learning_rate=LEARNR_VAE, betas=BETA_VAE,
                                                                is_save=True)

            combined_task_branch_synthetic.create_and_train_CNN(model_id=name, num_epochs=EPOCH_CNN, batch_size=BATCH,
                                                                is_frozen=False,
                                                                is_off_shelf_model=True,
                                                                epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                learning_rate=LEARNR_CNN,
                                                                betas=BETA_CNN, is_save=is_save)

            # perform test set benchmarks and record to file
            given_test_set_compare_synthetic_and_normal_approaches([combined_task_branch_synthetic], task_DIS_orig,category_list_subset, extra_new_cat_multi)
            record_keeper.record_to_file("real_versus fake continual learning adding "+str(category_list_subset[0])+" to "+str(category_list_subset[-1])+" synth multi "+str(extra_new_cat_multi))

            # delete from GPU and empty from cache
            del task_DIS_orig
            torch.cuda.empty_cache()


        # after the intial training, now perform real versus synthetic tests
        else:


            # TRAIN USING REAL SAMPLES
            print("\n------------ Real Samples", category_list_subset)
            combined_task_branch_no_synthetic = task.TaskBranch(task_DIS_orig.name +"real ["+str(category_list_subset[0])+" to "+str(category_list_subset[-1])+ "] ",task_DIS_orig, device, PATH_MODELS, record_keeper)

            # TRAIN REAL CNN
            print("\nReal Samples - CNN")
            combined_task_branch_no_synthetic.create_and_train_CNN(model_id=name, num_epochs=EPOCH_CNN,
                                                                   batch_size=BATCH, is_frozen=False,
                                                                   is_off_shelf_model=True,
                                                                   epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                   learning_rate=LEARNR_CNN,
                                                                   betas=BETA_CNN, is_save=False)
            # TRAIN REAL VAE
            print("\nReal Samples - VAE")
            combined_task_branch_no_synthetic.create_and_train_VAE(model_id=name, num_epochs=EPOCH_VAE, hidden_dim=10,
                                                                   latent_dim=LAT_DIM_VAE,
                                                                   is_synthetic=False,is_take_existing_VAE=False, is_new_categories_to_addded_to_existing_task=False,
                                                                   epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                   learning_rate=LEARNR_VAE,
                                                                   betas=BETA_VAE, is_save=False, batch_size=BATCH)

            # TRAIN USING FAKE SAMPLES
            print("\n------------ Pseudo Samples ",category_list_subset)


            # GENERATE THE SAMPLES AND BLEND WITH THE NEW CATEGORY
            print("\nPseudo Samples - create samples")
            combined_task_branch_synthetic.create_blended_dataset_with_synthetic_samples(task_DIS_orig,[category_list_subset[-1]],extra_new_cat_multi)


            # TRAIN PSEUDO/REAL VAE
            # note we use transfer learning and take the prior VAE
            print("\nPseudo Samples - Training VAE")
            combined_task_branch_synthetic.create_and_train_VAE(model_id=name, num_epochs=EPOCH_VAE, batch_size=BATCH,
                                                                hidden_dim=10,
                                                                latent_dim=LAT_DIM_VAE,
                                                                is_synthetic=False, is_take_existing_VAE=True,
                                                                teacher_VAE=combined_task_branch_synthetic.VAE_most_recent,
                                                                is_new_categories_to_addded_to_existing_task=True,
                                                                is_completely_new_task=False,
                                                                epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                learning_rate=LEARNR_VAE, betas=BETA_VAE,
                                                                is_save=is_save)
            # TRAIN PSEUDO/REAL CNN
            print("\nPseudo Samples - CNN")
            combined_task_branch_synthetic.create_and_train_CNN(model_id=name, num_epochs=EPOCH_CNN, batch_size=BATCH,
                                                                is_frozen=False,
                                                                is_off_shelf_model=True,
                                                                epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                learning_rate=LEARNR_CNN,
                                                                betas=BETA_CNN, is_save=is_save)


            # COMPARE REAL AND FAKE ON TEST SET, AND RECORD TO FILE
            given_test_set_compare_synthetic_and_normal_approaches([combined_task_branch_synthetic, combined_task_branch_no_synthetic], task_DIS_orig,
                                                                   category_list_subset, extra_new_cat_multi)
            record_keeper.record_to_file("real_versus fake continual learning adding "+str(category_list_subset)+" synth multi "+str(extra_new_cat_multi))

            # delete from GPU and empty from cache
            del task_DIS_orig, combined_task_branch_no_synthetic
            torch.cuda.empty_cache()


def test_synthetic_samples_versus_normal_increasing_PRETRAINED_VAE(original_task_datasetInterface,PATH_MODELS,model_string_vae, model_string_cnn, record_keeper,
                                         device='cuda',number_increments=10, extra_new_cat_multi=1):

    """Performs the test set evaluations from 'test_synthetic_samples_versus_normal_increasing' but with previously trained/saved models """

    print("***** Testing pseudo-rehearsal versus real samples")

    cat_list_all = original_task_datasetInterface.categories_list

    combined_task_branch_synthetic = None

    # Increase categories one by one and train data sets.
    # Identical to 'test_synthetic_samples_versus_normal_increasing' but loads pre-trained models instead of training them
    for increment in range(0, number_increments):

        task_DIS_orig = copy.deepcopy(original_task_datasetInterface)
        category_list_subset = cat_list_all[0:increment+1]
        task_DIS_orig.reduce_categories_in_dataset(category_list_subset)
        task_DIS_orig_for_synth = copy.deepcopy(task_DIS_orig)


        if (increment == 0):
            combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig_for_synth.name + " pseudo", task_DIS_orig_for_synth,
                                                             device, PATH_MODELS,record_keeper)

            print("Starting point, just MNIST....")
            print("Training VAE - Starting")

            print("\n Beginning point only ",category_list_subset)
            combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS + model_string_vae[increment])
            combined_task_branch_synthetic.num_categories_in_task = increment+1

            print("\nPseudo Samples - CNN")
            combined_task_branch_synthetic.load_existing_CNN(PATH_MODELS + model_string_cnn[increment])

            given_test_set_compare_synthetic_and_normal_approaches([combined_task_branch_synthetic], task_DIS_orig,category_list_subset, extra_new_cat_multi)

            record_keeper.record_to_file("only blur post:real_versus fake continual learning adding "+str(category_list_subset[0])+" to "+str(category_list_subset[-1])+" synth multi "+str(extra_new_cat_multi))

            del task_DIS_orig
            torch.cuda.empty_cache()

        else:

            print("\n------------ Pseudo Samples ",category_list_subset)

            print("\nPseudo Samples - Loading VAE")
            combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS + model_string_vae[increment])
            combined_task_branch_synthetic.num_categories_in_task = increment+1

            print("\nPseudo Samples - CNN")
            combined_task_branch_synthetic.load_existing_CNN(PATH_MODELS + model_string_cnn[increment])

            given_test_set_compare_synthetic_and_normal_approaches([combined_task_branch_synthetic], task_DIS_orig,category_list_subset, extra_new_cat_multi)

            record_keeper.record_to_file("only blur post:real_versus fake continual learning adding "+str(category_list_subset)+" synth multi "+str(extra_new_cat_multi))

            del task_DIS_orig
            torch.cuda.empty_cache()


def given_test_set_compare_synthetic_and_normal_approaches(task_branch_list, dataSetInter,new_cats_added,extra_new_cat_multi):

    """Helper function that evaluates a list of task branch on a given datasets test split"""


    for task_branch in task_branch_list:

        print("\n_____________No blurring")
        task_branch.run_end_of_training_benchmarks("real_versus fake continual learning adding "+str(new_cats_added)+" synth multi "+str(extra_new_cat_multi),dataSetInter,is_gaussian_noise_required=False, is_save=False)
        print("\n_____________Blurring")
        task_branch.run_end_of_training_benchmarks("real_versus fake continual learning adding "+str(new_cats_added)+" synth multi "+str(extra_new_cat_multi),dataSetInter,is_gaussian_noise_required=True, is_save=False)
        print("\n_____________Blurring+ best select")
        task_branch.run_end_of_training_benchmarks("real_versus fake continual learning adding "+str(new_cats_added)+" synth multi "+str(extra_new_cat_multi),dataSetInter,is_gaussian_noise_required=True, is_extra_top_three_method=True, is_save=False)



def test_concept_drift_for_single_task(task_branch, shear_degree_max,shear_degree_increments, split, num_samples_to_check=100):

        """Tests how well the Agent can identify concept drift, and then retrain accordingly. This is compared against an Agent that does not retrain.
         A affine/shear transform is used as a proxy for concept drift"""

        print("*** Testing how task relativity changes with concept drift: ", task_branch.task_name)

        # initialise variables
        shear_degree_increment_num  = round(shear_degree_max/shear_degree_increments)
        task_branch_no_recalibration_changes = copy.deepcopy(task_branch)
        task_branch = copy.deepcopy(task_branch)
        task_branch_no_recalibration_changes.name = "No recalibration changes"

        is_save = True
        BATCH = 64
        EPOCH_IMPROVE_LIMIT = 20
        EPOCH_CNN = 3
        BETA_CNN = (0.999, .999)
        LEARNR_CNN = 0.00025
        EPOCH_VAE = 10
        LAT_DIM_VAE = 50
        BETA_VAE = (0.5, 0.999)
        LEARNR_VAE = 0.00035

        recal_count = 0

        # gradually increase the shear transform
        for increment in range(0,shear_degree_increment_num+1):
            shear_degree = increment*shear_degree_increments
            shear_trans = transforms.Compose([lambda img: transforms.functional.affine(img, angle=0, translate=(0, 0),scale=1, shear=shear_degree)])

            # update the transformations so data is sheared when drawn
            task_branch.dataset_interface.update_transformations(shear_trans)
            task_branch_no_recalibration_changes.dataset_interface.update_transformations(shear_trans)
            dataloader = task_branch.dataset_interface.return_data_loaders(branch_component='VAE', BATCH_SIZE=1)[split]

            # find task relatedness for task branch with recalibration
            reconstruction_error_average, task_relatedness = task_branch.given_task_data_set_find_task_relatedness(
                dataloader, num_samples_to_check=num_samples_to_check)

            dataloader = task_branch.dataset_interface.return_data_loaders(branch_component='VAE', BATCH_SIZE=1)[split]
            task_branch_no_recalibration_changes.reset_queue_variables()

            # find task relatedness for task branch WITHOUT recalibration
            reconstruction_error_average_no_recal, task_relatedness_no_recal = task_branch_no_recalibration_changes.given_task_data_set_find_task_relatedness(dataloader, num_samples_to_check=num_samples_to_check)

            print("   --- With shear degree:", shear_degree)
            print("   --- With Recal: Reconstruction error average", reconstruction_error_average, " Task relatedness",task_relatedness, "Number of samples:", num_samples_to_check)
            print("   --- No Recal:   Reconstruction error average", reconstruction_error_average_no_recal, " Task relatedness",task_relatedness_no_recal, "Number of samples:", num_samples_to_check)


            # Check if recalibration has been triggered, and if so, retrain VAE and CNN with pseudo-samples
            if task_branch.is_need_to_recalibrate_queue:
                print("   --- VAE quality has fallen below threshold of",task_branch.task_relatedness_threshold)
                print("   --- Recalibrate with newer samples and pseudo samples representing older distribution")
                task_branch.create_blended_dataset_with_synthetic_samples(copy.deepcopy(task_branch.dataset_interface), [], extra_new_cat_multi=1.0, is_single_task_recal_process= True)

                print("\nPseudo Samples - Training VAE")
                task_branch.create_and_train_VAE(model_id="recalibrated wth shear "+str(shear_degree), num_epochs=EPOCH_VAE,
                                                                        batch_size=BATCH,
                                                                        hidden_dim=10,
                                                                        latent_dim=LAT_DIM_VAE,
                                                                        is_synthetic=False, is_take_existing_VAE=True,
                                                                        teacher_VAE=task_branch.VAE_most_recent,
                                                                        is_new_categories_to_addded_to_existing_task=False,
                                                                        is_completely_new_task=False,
                                                                        epoch_improvement_limit=20,
                                                                        learning_rate=LEARNR_VAE, betas=BETA_VAE,sample_limit=num_samples_to_check,
                                                                        is_save=is_save)
                print("\nPseudo Samples - CNN")
                task_branch.create_and_train_CNN(model_id="recalibrated wth shear "+str(shear_degree), num_epochs=EPOCH_CNN,
                                                                        batch_size=BATCH,
                                                                        is_frozen=False,
                                                                        is_off_shelf_model=True,
                                                                        epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                        learning_rate=LEARNR_CNN,sample_limit=num_samples_to_check,
                                                                        betas=BETA_CNN, is_save=is_save)
                recal_count += 1

            # calculate benchmarks for recalibration versus no-recalibration
            print("\nACCURACY - RECALIBRATION: ",recal_count," times")
            task_branch.run_end_of_training_benchmarks("recalibration vs no recalibration", task_branch.dataset_interface)

            print("\nACCURACY - NO RECALIBRATION")
            task_branch_no_recalibration_changes.run_end_of_training_benchmarks("recalibration vs no recalibration", task_branch.dataset_interface)



def load_VAE_models_and_display_syn_images(PATH_MODELS,model_string, task_branch):


    """Helper method that displays images from pre-trained VAE model/models"""

    task_branch.VAE_most_recent = None
    task_branch.CNN_most_recent = None


    FOLDER = "real_versus_synth_models/"

    i=1
    for string in model_string:
        task_branch.load_existing_VAE(PATH_MODELS+FOLDER+string)
        task_branch.generate_samples_to_display()
        task_branch.num_categories_in_task = i
        i += 1

    task_branch.run_end_of_training_benchmarks("double check", is_save=False, is_gaussian_noise_required=True)


def distance_calculation(*task_branches, num_per_domain = 10):

    """Helper function. Takes random images from each dataset and calculates the Euclidean distance between them"""

    list_images_all = []

    dist = np.zeros((len(task_branches),len(task_branches)))
    dist_recon = np.zeros((len(task_branches),len(task_branches)))

    # perform the random sampling several times
    for i in range(0, num_per_domain):

        # calculate sample distance between each of the domains
        for i in range(0, len(task_branches)):
            for j in range(0, len(task_branches)):
                task1 = task_branches[i]
                task2 = task_branches[j]
                task1.VAE_most_recent.send_all_to_GPU()
                task2.VAE_most_recent.send_all_to_GPU()

                rand_int1 = random.randint(0,task1.dataset_interface.training_set_size-1)
                rand_int2 = random.randint(0,task2.dataset_interface.training_set_size-1)
                rand_image1,cat1 = task1.dataset_interface.dataset['train']['VAE'].__getitem__(rand_int1)
                rand_image2,cat2 = task2.dataset_interface.dataset['train']['VAE'].__getitem__(rand_int2)
                dist_single = np.sqrt(np.sum((rand_image1.numpy()-rand_image2.numpy())**2))


                dist[i,j] += dist_single

                # calculate the same distances after reconstruction from VAE

                cat1 = idx2onehot(torch.tensor([[cat1]]), task1.num_categories_in_task)
                cat1 = cat1.to('cuda')

                cat2 = idx2onehot(torch.tensor([[cat2]]), task2.num_categories_in_task)
                cat2 = cat2.to('cuda')

                rand_image1_recon,_,_ = task1.VAE_most_recent.encode_then_decode_without_randomness(rand_image1.to('cuda'),cat1)
                rand_image2_recon,_,_ = task2.VAE_most_recent.encode_then_decode_without_randomness(rand_image2.to('cuda'),cat2)


                rand_image1_recon = rand_image1_recon.cpu().detach().numpy()
                rand_image2_recon = rand_image2_recon.cpu().detach().numpy()

                dist_single_recon = np.sqrt(np.sum(((rand_image1_recon-rand_image2_recon)**2)))
                dist_recon[i,j] += dist_single_recon

    print("no recon")
    print(dist)
    print(dist/num_per_domain)

    print("with recon")
    print(dist_recon)
    print(dist_recon/num_per_domain)




def pass_through_images_in_vaes(*task_branches, num_per_domain = 10):

    """Helper function. Randomly selects a sample, and passes it through each task's VAE"""

    for i in range(0, num_per_domain):
        fig2 = plt.figure(figsize=(15, 15))

        image_count =1
        for i in range(0, len(task_branches)):
            task1 = task_branches[i]
            rand_int1 = random.randint(0,task1.dataset_interface.training_set_size-1)
            rand_image1,cat1 = task1.dataset_interface.dataset['train']['VAE'].__getitem__(rand_int1)

            ax = fig2.add_subplot(len(task_branches), len(task_branches) + 1, image_count)
            ax.axis('off')
            ax.set_title("Original "+task1.task_name)
            ax.imshow(torch.squeeze(rand_image1), cmap='gray')
            image_count += 1

            for j in range(0, len(task_branches)):
                task2 = task_branches[j]
                task1.VAE_most_recent.send_all_to_GPU()
                task2.VAE_most_recent.send_all_to_GPU()

                rand_image1 = rand_image1.to('cuda')

                info, rand_image1_recon = task2.VAE_most_recent.get_sample_reconstruction_error_from_all_category(rand_image1, by_category_mean_std_of_reconstruction_error=None, is_random = False, only_return_best = True, is_standardised_distance_check = False)


                rand_image1_recon = torch.squeeze(rand_image1_recon).cpu().detach().numpy()
                dist_single = np.sqrt(np.sum((rand_image1.cpu().detach().numpy() - rand_image1_recon) ** 2))

                ax = fig2.add_subplot(len(task_branches), len(task_branches) + 1, image_count)
                ax.axis('off')
                ax.set_title(task2.task_name + f'\n Euclidean dist {dist_single:.3f}\n Recon err {info[0][1]:.3f}')
                ax.imshow(rand_image1_recon, cmap='gray')
                image_count += 1

        plt.ioff()
        plt.show()

    print("\n\n")

