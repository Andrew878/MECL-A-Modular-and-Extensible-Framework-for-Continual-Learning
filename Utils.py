import torch.nn as nn
import torch
import TaskBranch as task
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
    # print( n)
    assert torch.max(idx).item() < num_categories

    onehot = torch.zeros(idx.size(0), num_categories)
    #print(onehot)
    onehot.scatter_(1, idx.data, 1)

    return onehot


def generate_list_of_class_category_tensors(N_CLASSES):
    class_list = []
    for j in range(0, N_CLASSES):
        class_list.append(torch.tensor(np.array([[j], ])).to(dtype=torch.long))
    return class_list


def test_generating_and_classification_ability_multi_tasks(task_branches, number_per_category=1, device='cuda'):
    print("\n************ Testing generation and classification matches:")
    for task_branch in task_branches:
        correct_matches_by_class = test_generating_and_classification_ability_single_task(number_per_category,
                                                                                          task_branch)
        test_generator_accuracy(correct_matches_by_class, number_per_category)


def test_generating_and_classification_ability_single_task(number_per_category, task_branch, device='cuda'):
    print("\nTask: ", task_branch.task_name)
    correct_matches_by_class = {i: 0 for i in range(0, task_branch.num_categories_in_task)}
    synthetic_data_list_x, synthetic_data_list_y, = task_branch.VAE_most_recent.generate_synthetic_set_all_cats(
        number_per_category=number_per_category)

    for i in range(0, len(synthetic_data_list_x)):

        img_transformed = task_branch.dataset_interface.transformations['CNN']['test_to_image'](
            synthetic_data_list_x[i].cpu()).float()

        # None is because model is expecting a batch, and we want only a single observation. Also need to send to CUDA
        pred = task_branch.classify_with_CNN(img_transformed[None].to(device))
        if (pred.item() == synthetic_data_list_y[i]):
            correct_matches_by_class[synthetic_data_list_y[i]] += 1

    return correct_matches_by_class


def test_generator_accuracy(accuracy_score_by_class, number_per_category):
    total = 0
    for category in accuracy_score_by_class:
        total += accuracy_score_by_class[category]
        accuracy = 1.0 * accuracy_score_by_class[category] / (1.0 * number_per_category)
        print('Class: {}, % of VAE-CNN matches: {:.00%} out of {}'.format(category, accuracy, number_per_category))
    total_accuracy = total / (len(accuracy_score_by_class) * number_per_category)
    print('All classes overall, % of VAE-CNN matches: {:%} out of {}'.format(total_accuracy, (
        len(accuracy_score_by_class)) * number_per_category))


def test_gate_allocation(gate, *data_set_interfaces, number_tests_per_data_set=1000):
    overall_correct_task_allocations_recon = 0
    overall_correct_task_allocations_std = 0

    overall_correct_task_and_class_allocations_recon = 0
    overall_correct_task_and_class_allocations_std = 0

    num_tests_overall = 0

    for data_set_interface in data_set_interfaces:

        print("\n ********  For samples taken from ", data_set_interface.name)

        data_loaders = data_set_interface.return_data_loaders(branch_component='VAE', BATCH_SIZE=1)
        num_tests = 0

        correct_task_allocation_recon = 0
        correct_task_and_class_allocation_recon = 0

        correct_task_allocation_std = 0
        correct_task_and_class_allocation_std = 0

        for i, (x, y) in enumerate(data_loaders['val']):

            #            print("actual", y)
            best_task_recon, best_class_recon, best_task_std, best_class_std = gate.allocate_sample_to_task_branch(x,
                                                                                                                   is_standardised_distance_check=True,
                                                                                                                   is_return_both_metrics=True)

            if (best_task_recon.task_name == data_set_interface.name):
                correct_task_allocation_recon += 1
                if (best_class_recon == y.item()):
                    correct_task_and_class_allocation_recon += 1
            # else:
            #     print("actual",y.item(),data_set_interface.name, "guess", best_task_recon, best_class_recon)

            if (best_task_std.task_name == data_set_interface.name):
                correct_task_allocation_std += 1
                if (best_class_std == y.item()):
                    correct_task_and_class_allocation_std += 1
            # else:
            #     print("actual",y.item(),data_set_interface.name, "guess", best_task_std.task_name, best_class_std)

            num_tests += 1

            if number_tests_per_data_set <= num_tests:
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
                                overall_correct_task_and_class_allocations_recon,
                                overall_correct_task_and_class_allocations_std,
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


def test_pre_trained_versus_non_pre_trained(new_task_to_be_trained, template_task, model_id, num_epochs=30,
                                            batch_size=64, hidden_dim=10, latent_dim=75, epoch_improvement_limit=20,
                                            learning_rate=0.00035, betas=(0.5, .999), is_save=False, ):
    print("--- Task to be trained: ", new_task_to_be_trained.task_name)
    print("*********** Training from scratch ")

    new_task_to_be_trained.create_and_train_VAE(model_id=model_id + "untrained", num_epochs=num_epochs,
                                                batch_size=batch_size,
                                                hidden_dim=10,
                                                latent_dim=latent_dim, is_take_existing_VAE=True,
                                                teacher_VAE=template_task.VAE_most_recent, is_completely_new_task=True,
                                                epoch_improvement_limit=epoch_improvement_limit,
                                                learning_rate=learning_rate, betas=betas, is_save=is_save)

    print("*********** Training with pretrained weights from: ", template_task.task_name)

    new_task_to_be_trained.create_and_train_VAE(model_id=model_id + "pretrained", num_epochs=num_epochs,
                                                batch_size=batch_size, hidden_dim=10,
                                                latent_dim=latent_dim, is_take_existing_VAE=False,
                                                teacher_VAE=None, is_completely_new_task=True,
                                                epoch_improvement_limit=epoch_improvement_limit,
                                                learning_rate=learning_rate, betas=betas, is_save=is_save)


def test_synthetic_samples_versus_normal(original_task_datasetInterface, added_task_datasetInterface, PATH_MODELS,
                                         device='cuda', new_classes_per_increment=1, number_increments=3 , extra_new_cat_multi=1):
    print("***** Testing pseudo-rehearsal versus real samples")

    new_class_index = 26
    is_time_to_break = False

    is_save = False
    BATCH = 64
    EPOCH_IMPROVE_LIMIT = 20
    num_per_cat_gen_class_test = 1
    num_per_cat_gen_class_test = 1000

    EPOCH_CNN = 1
    EPOCH_CNN = 5
    BETA_CNN = (0.999, .999)
    LEARNR_CNN = 0.00025

    EPOCH_VAE = 2
    EPOCH_VAE = 30
    LAT_DIM_VAE = 50
    BETA_VAE = (0.5, 0.999)
    LEARNR_VAE = 0.00035

    task_DIS_orig_for_synthetic = copy.deepcopy(original_task_datasetInterface)
    task_DIS_New_for_synthetic = copy.deepcopy(added_task_datasetInterface)

    combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig_for_synthetic.name + " pseudo", task_DIS_orig_for_synthetic,
                                                     device, PATH_MODELS)

    for increment in range(0, number_increments):

        task_DIS_orig = copy.deepcopy(original_task_datasetInterface)
        task_DIS_added = copy.deepcopy(added_task_datasetInterface)

        name = "increment" + str(increment)

        if (increment == 0):
            print("Starting point, just MNIST....")
            print("Training VAE - Starting")
            # combined_task_branch_synthetic.create_and_train_VAE(model_id=name, num_epochs=EPOCH_VAE, batch_size=BATCH,
            #                                                     hidden_dim=10,
            #                                                     latent_dim=LAT_DIM_VAE,
            #                                                     is_synthetic=False, is_take_existing_VAE=False,
            #                                                     teacher_VAE=None,
            #                                                     is_new_categories_to_addded_to_existing_task=False,
            #                                                     is_completely_new_task=True,
            #                                                     epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
            #                                                     learning_rate=LEARNR_VAE, betas=BETA_VAE,
            #                                                     is_save=True)

            combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS+"VAE MNIST pseudo epochs2,batch128,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 103.26511754150391 increment0",False)

        else:


            new_class_index -= new_classes_per_increment
            new_class_index = min(new_class_index, task_DIS_added.num_categories)

            # get list of new labels to add
            list_categories_to_add = copy.deepcopy(added_task_datasetInterface.categories_list[new_class_index:len(added_task_datasetInterface.categories_list)])
            list_categories_to_add.reverse()
            list_categories_to_add_marginal = copy.deepcopy([added_task_datasetInterface.categories_list[new_class_index]])
            print("\n\n_________________________________________________________________________\nlist_categories_to_add_marginal",list_categories_to_add_marginal)

            # merge dataset
            task_DIS_orig.add_outside_data_to_data_set(task_DIS_added, list_categories_to_add)

            combined_task_branch_no_synthetic = task.TaskBranch(task_DIS_orig.name + str(list_categories_to_add) +" real",
                                                                task_DIS_orig, device, PATH_MODELS)

            print("\n------------ Real Samples", list_categories_to_add)
            print("\nReal Samples - CNN")
            combined_task_branch_no_synthetic.create_and_train_CNN(model_id=name, num_epochs=EPOCH_CNN,
                                                                   batch_size=BATCH, is_frozen=False,
                                                                   is_off_shelf_model=True,
                                                                   epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                   learning_rate=LEARNR_CNN,
                                                                   betas=BETA_CNN, is_save=is_save)
            print("\nReal Samples - VAE")
            combined_task_branch_no_synthetic.create_and_train_VAE(model_id=name, num_epochs=EPOCH_VAE, hidden_dim=10,
                                                                   latent_dim=LAT_DIM_VAE,
                                                                   is_synthetic=False,is_take_existing_VAE=False, is_new_categories_to_addded_to_existing_task=False,
                                                                   epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                   learning_rate=LEARNR_VAE,
                                                                   betas=BETA_VAE, is_save=is_save, batch_size=BATCH)

            print("\n------------ Pseudo Samples", list_categories_to_add)

            print("\nPseudo Samples - create samples")
            combined_task_branch_synthetic.create_blended_dataset_with_synthetic_samples(task_DIS_New_for_synthetic,
                                                                                         list_categories_to_add_marginal,extra_new_cat_multi)

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
            print("\nPseudo Samples - CNN")
            combined_task_branch_synthetic.create_and_train_CNN(model_id=name, num_epochs=EPOCH_CNN, batch_size=BATCH,
                                                                is_frozen=False,
                                                                is_off_shelf_model=True,
                                                                epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                learning_rate=LEARNR_CNN,
                                                                betas=BETA_CNN, is_save=is_save)



            given_test_set_compare_synthetic_and_normal_approaches(combined_task_branch_synthetic, combined_task_branch_no_synthetic, task_DIS_orig,
                                                                   list_categories_to_add)

            test_generating_and_classification_ability_multi_tasks([combined_task_branch_synthetic,combined_task_branch_no_synthetic], number_per_category=num_per_cat_gen_class_test, device=device)


def given_test_set_compare_synthetic_and_normal_approaches(task_branch_synth, task_branch_real, dataSetInter,new_cats_added):

    device = 'cuda'

    for task_branch in [task_branch_synth, task_branch_real]:

        print("\n *************\nFor task ", task_branch.task_name, new_cats_added)

        data_loader_CNN = dataSetInter.return_data_loaders('CNN', BATCH_SIZE = 1)
        data_loader_VAE = dataSetInter.return_data_loaders('VAE', BATCH_SIZE = 1)


        task_branch.VAE_most_recent.eval()
        task_branch.CNN_most_recent.eval()
        task_branch.VAE_most_recent.to(device)
        task_branch.CNN_most_recent.to(device)

        # To update record for mean and std deviation distance. This deletes old entries before calculating
        by_category_record_of_recon_error_and_accuracy = {i: [0,0,0] for i in range(0, task_branch.num_categories_in_task)}



        for i, (x, y) in enumerate(data_loader_VAE['val']):
            x = x.to(device)
            y_original = y.item()
            y = idx2onehot(y.view(-1, 1), task_branch.num_categories_in_task)
            y = y.to(device)

            with torch.no_grad():
                reconstructed_x, z_mu, z_var = task_branch.VAE_most_recent(x, y)

            # loss
            loss = task_branch.VAE_most_recent.loss(x, reconstructed_x, z_mu, z_var)

            by_category_record_of_recon_error_and_accuracy[y_original][0] += 1
            by_category_record_of_recon_error_and_accuracy[y_original][1] += loss.item()


        for i, (x, y) in enumerate(data_loader_CNN['val']):
            # reshape the data into [batch_size, 784]
            # print(x.size())
            # x = x.view(batch_size, 1, 28, 28)
            x = x.to(device)
            y_original = y.item()
            y = y.to(device)

            with torch.no_grad():
                outputs = task_branch.CNN_most_recent(x)
                _, preds = torch.max(outputs, 1)
                # print(preds)
                #criterion = nn.CrossEntropyLoss()
                #loss = criterion(outputs.to(device), y)

                correct = torch.sum(preds == y.data)

            by_category_record_of_recon_error_and_accuracy[y_original][2] += correct.item()

        task_branch.VAE_most_recent.cpu()
        task_branch.CNN_most_recent.cpu()


        total_count =0
        total_recon =0
        total_correct =0
        for category in by_category_record_of_recon_error_and_accuracy:

            count = by_category_record_of_recon_error_and_accuracy[category][0]
            recon_ave = by_category_record_of_recon_error_and_accuracy[category][1]/count
            accuracy = by_category_record_of_recon_error_and_accuracy[category][2] /count

            total_count += by_category_record_of_recon_error_and_accuracy[category][0]
            total_recon += by_category_record_of_recon_error_and_accuracy[category][1]
            total_correct += by_category_record_of_recon_error_and_accuracy[category][2]

            print("For:",count, category," Ave. Recon:",recon_ave," Ave. Accuracy:",accuracy)

        print("For all (",total_count,"): Ave. Recon:",total_recon/total_count," Ave. Accuracy:",total_correct/total_count)

