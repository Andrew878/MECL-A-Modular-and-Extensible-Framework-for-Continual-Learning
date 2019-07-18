import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import max, zeros
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def idx2onehot(idx, num_categories):
    """Returns a one_hot_encoded_vector Borrowed from xxxx"""

    assert idx.shape[1] == 1
    #print( n)
    assert torch.max(idx).item() < num_categories

    onehot = torch.zeros(idx.size(0), num_categories)
    onehot.scatter_(1, idx.data, 1)

    return onehot


def generate_list_of_class_category_tensors(N_CLASSES):
    class_list = []
    for j in range(0, N_CLASSES):
        class_list.append(torch.tensor(np.array([[j], ])).to(dtype=torch.long))
    return class_list


def test_generating_and_classification_ability_multi_tasks(task_branches, number_per_category = 1, device = 'cuda'):

    print("Testing generation and classification matches:")
    for task_branch in task_branches:

        correct_matches_by_class = test_generating_and_classification_ability_single_task(number_per_category, task_branch)
        test_generator_accuracy(correct_matches_by_class, number_per_category)



def test_generating_and_classification_ability_single_task(number_per_category, task_branch,device = 'cuda'):
    print("\nTask: ", task_branch.task_name)
    correct_matches_by_class = {i: 0 for i in range(0, task_branch.num_categories_in_task)}
    synthetic_data_list_x, synthetic_data_list_y, = task_branch.VAE_most_recent.generate_synthetic_set_all_cats(number_per_category=number_per_category)

    for i in range(0, len(synthetic_data_list_x)):

        img_transformed = task_branch.dataset_interface.transformations['CNN']['test_to_image'](synthetic_data_list_x[i].cpu()).float()

        # None is because model is expecting a batch, and we want only a single observation. Also need to send to CUDA
        pred = task_branch.classify_with_CNN(img_transformed[None].to(device))
        if (pred.item() == synthetic_data_list_y[i]):

            correct_matches_by_class[synthetic_data_list_y[i]] += 1

    return  correct_matches_by_class

def test_generator_accuracy(accuracy_score_by_class, number_per_category):
    total = 0
    for category in accuracy_score_by_class:
        total += accuracy_score_by_class[category]
        accuracy = 1.0 * accuracy_score_by_class[category] / (1.0 * number_per_category)
        print('Class: {}, % of VAE-CNN matches: {:.00%} out of {}'.format(category, accuracy, number_per_category))
    total_accuracy = total / (len(accuracy_score_by_class)* number_per_category)
    print('All classes overall, % of VAE-CNN matches: {:%} out of {}'.format(total_accuracy, (
        len(accuracy_score_by_class)) * number_per_category))


def test_gate_allocation(gate, *data_set_interfaces, number_tests_per_data_set = 1000):

    overall_correct_task_allocations_recon = 0
    overall_correct_task_allocations_std = 0

    overall_correct_task_and_class_allocations_recon = 0
    overall_correct_task_and_class_allocations_std = 0

    num_tests_overall = 0

    for data_set_interface in data_set_interfaces:

        print("\n ********  For samples taken from ", data_set_interface.name)


        data_loaders = data_set_interface.return_data_loaders(branch_component='VAE', BATCH_SIZE = 1)
        num_tests = 0

        correct_task_allocation_recon = 0
        correct_task_and_class_allocation_recon = 0

        correct_task_allocation_std = 0
        correct_task_and_class_allocation_std = 0

        for i, (x, y) in enumerate(data_loaders['val']):

#            print("actual", y)
            best_task_recon, best_class_recon,best_task_std, best_class_std = gate.allocate_sample_to_task_branch(x, is_standardised_distance_check=True, is_return_both_metrics=True)

            if(best_task_recon.task_name == data_set_interface.name):
                correct_task_allocation_recon += 1
                if(best_class_recon == y.item()):
                    correct_task_and_class_allocation_recon +=1
            # else:
            #     print("actual",y.item(),data_set_interface.name, "guess", best_task_recon, best_class_recon)

            if(best_task_std.task_name == data_set_interface.name):
                correct_task_allocation_std += 1
                if(best_class_std == y.item()):
                    correct_task_and_class_allocation_std +=1
            # else:
            #     print("actual",y.item(),data_set_interface.name, "guess", best_task_std.task_name, best_class_std)

            num_tests += 1

            if number_tests_per_data_set <= num_tests :
                break

        overall_correct_task_allocations_recon += correct_task_allocation_recon
        overall_correct_task_allocations_std += correct_task_allocation_std

        overall_correct_task_and_class_allocations_recon += correct_task_and_class_allocation_recon
        overall_correct_task_and_class_allocations_std += correct_task_and_class_allocation_std
        num_tests_overall += num_tests

        gate_accuracy_print_results(correct_task_allocation_recon, correct_task_allocation_std,
                                    correct_task_and_class_allocation_recon, correct_task_and_class_allocation_std,
                                    num_tests)
    print("\n ********  Overall Results")
    gate_accuracy_print_results(overall_correct_task_allocations_recon, overall_correct_task_allocations_std,
                                overall_correct_task_and_class_allocations_recon, overall_correct_task_and_class_allocations_std,
                                num_tests_overall)


def gate_accuracy_print_results(correct_task_allocation_recon, correct_task_allocation_std,
                                correct_task_and_class_allocation_recon, correct_task_and_class_allocation_std,
                                num_tests):
    print("Out of", num_tests, "samples.")
    print(
        'Reconstruction error:   {} correct, % of correct allocations: {:.00%}, % of correct allocations and class: {:.00%}'.format(
            correct_task_allocation_recon, correct_task_allocation_recon / num_tests,
            correct_task_and_class_allocation_recon / num_tests))
    print(
        'Std dev distance error: {} correct, % of correct allocations: {:.00%}, % of correct allocations and class: {:.00%}'.format(
            correct_task_allocation_std, correct_task_allocation_std / num_tests,
            correct_task_and_class_allocation_std / num_tests))


def test_pre_trained_versus_non_pre_trained(new_task_to_be_trained, template_task, model_id, num_epochs=30, batch_size=64, hidden_dim=10, latent_dim=75, epoch_improvement_limit=20, learning_rate=0.00035, betas=(0.5, .999), is_save=False,):

    print("--- Task to be trained: ",new_task_to_be_trained.task_name)
    print("*********** Training from scratch ")

    new_task_to_be_trained.create_and_train_VAE(model_id=model_id+"untrained", num_epochs=num_epochs, batch_size=batch_size,
                                                hidden_dim=10,
                                                latent_dim=latent_dim, is_take_existing_VAE=True,
                                                teacher_VAE=template_task.VAE_most_recent, is_completely_new_task=True,
                                                epoch_improvement_limit=epoch_improvement_limit,
                                                learning_rate=learning_rate, betas=betas, is_save=is_save)

    print("*********** Training with pretrained weights from: ",template_task.task_name)
    new_task_to_be_trained.create_and_train_VAE(model_id=model_id+"pretrained", num_epochs=num_epochs, batch_size=batch_size, hidden_dim=10,
                                            latent_dim=latent_dim, is_take_existing_VAE=False,
                                            teacher_VAE=None, is_completely_new_task=True,
                                            epoch_improvement_limit=epoch_improvement_limit, learning_rate=learning_rate, betas=betas, is_save=is_save)



def test_synthetic_samples_versus_normal(original_task_datasetInterface, added_task_datasetInterface, new_classes_per_increment = 1, number_increments = 5):

    new_class_index = 0
    is_time_to_break = False

    for increment in range(0,number_increments):

        task_orig = copy.deepcopy(original_task_datasetInterface)
        task_added = copy.deepcopy(added_task_datasetInterface)


        if(increment != 0):
            new_class_index += new_classes_per_increment
            new_class_index = min(new_class_index,task_added.num_categories)


            list_categories_to_add = added_task_datasetInterface.categories_list[0:new_class_index+1]


    mnist_task_branch.create_blended_dataset_with_synthetic_samples(emnist_data_and_interface, list)
    print("\nTraining VAE for ", list)
    name = "mutation" + str(i)

    mnist_task_branch.create_and_train_VAE(model_id=name, num_epochs=30, batch_size=64, hidden_dim=10, latent_dim=50,
                                           is_synthetic=False, is_take_existing_VAE=True,
                                           teacher_VAE=mnist_task_branch.VAE_most_recent,
                                           is_new_categories_to_addded_to_existing_task=True,
                                           is_completely_new_task=False,
                                           epoch_improvement_limit=30, learning_rate=0.00035, betas=(0.5, .999),
                                           is_save=False, )

    print("\nTraining CNN for ", list)

    BATCH = 64
    mnist_task_branch.create_and_train_CNN(model_id=name, num_epochs=15, batch_size=BATCH, is_frozen=False,
                                           is_off_shelf_model=True, epoch_improvement_limit=20, learning_rate=0.00025,
                                           betas=(0.999, .999), is_save=True)
    i += 1
