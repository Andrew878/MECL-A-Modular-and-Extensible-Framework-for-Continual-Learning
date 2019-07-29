import torch.nn as nn
import torch
import TaskBranch as task
import copy
import Gate
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import max, zeros
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


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
        pred, _ = task_branch.classify_with_CNN(img_transformed[None].to(device))
        if (pred.item() == synthetic_data_list_y[i]):
            correct_matches_by_class[synthetic_data_list_y[i]] += 1

    # synthetic_data_list_x_short, synthetic_data_list_y_short, = task_branch.VAE_most_recent.generate_synthetic_set_all_cats(
    #     number_per_category=1)
    #
    # fig2 = plt.figure(figsize=(20, 20))
    # x = 0
    # r = 60
    # c = 5
    #
    # print("TRYING TO PRINT Z'S")
    # print(len(synthetic_data_list_x_short))
    # for i in range(x, r * c):
    #     img, cat = subset_dataset[i]
    #     img = img.view(28, 28).data
    #     img = img.numpy()
    #     ax = fig2.add_subplot(r, c, i - x + 1)
    #     ax.axis('off')
    #     ax.set_title(cat)
    #     ax.imshow(img, cmap='gray_r')
    #
    # plt.ioff()
    # plt.show()

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

def test_gate_versus_non_gate(*task_branches, number_tests_per_data_set=1000):


    print("*********** Testing Gate versus non-gate_method")
    gate = Gate.Gate()
    dataset_list = []
    confusion_matrix =  {}

    for task in task_branches:
        gate.add_task_branch(task)
        dataset_list.append(task.dataset_interface)
        confusion_matrix[task.task_name] = {}


    GATE_overall_correct_task_classification = 0
    NO_GATE_overall_correct_task_classification = 0

    per_task_GATE_overall_correct_task_classification = 0
    per_task_NO_GATE_overall_correct_task_classification = 0

    overall_correct_task_allocations_std = 0

    overall_correct_task_and_class_allocations_recon = 0
    overall_correct_task_and_class_allocations_std = 0

    num_tests_overall = 0
    per_task_num_tests = 0


    for data_set_interface in dataset_list:
        per_task_GATE_overall_correct_task_classification = 0
        per_task_NO_GATE_overall_correct_task_classification = 0
        per_task_num_tests = 0

        confusion_matrix[data_set_interface.name] = {task.task_name:0 for task in task_branches}

        dataloader = data_set_interface.return_data_loaders('VAE', BATCH_SIZE=1)

        for i, (x, y) in enumerate(dataloader['val']):

            num_tests_overall += 1
            per_task_num_tests += 1
            best_task_recon, pred_cat, __ = gate.classify_input_using_allocation_method(x)

            confusion_matrix[data_set_interface.name][best_task_recon.task_name] += 1

            if (best_task_recon.task_name == data_set_interface.name):
                if (pred_cat == y.item()):
                    GATE_overall_correct_task_classification += 1
                    per_task_GATE_overall_correct_task_classification +=1

            highest_prob = 0
            highest_prob_cat = 0
            highest_prob_task = 0
            for task in task_branches:
                x = data_set_interface.transformations['CNN']['test_to_image'](x)
                pred_cat, probability = task.classify_with_CNN(x)
                if probability>highest_prob:
                    highest_prob = probability
                    highest_prob_cat = pred_cat
                    highest_prob_task = task

            if (highest_prob_task.task_name == data_set_interface.name):
                if (highest_prob_cat == y.item()):
                    NO_GATE_overall_correct_task_classification += 1
                    per_task_NO_GATE_overall_correct_task_classification += 1

        print(data_set_interface.name," by task samples: ",per_task_num_tests, " Gate accuracy: ",per_task_GATE_overall_correct_task_classification/per_task_num_tests, " Accuracy without gate: ",per_task_NO_GATE_overall_correct_task_classification/per_task_num_tests)
    print("Overall Samples: ", num_tests_overall, " Gate accuracy: ",GATE_overall_correct_task_classification / num_tests_overall, " Accuracy without gate: ",NO_GATE_overall_correct_task_classification / num_tests_overall,"\n\n")

    print("*** Printing confusion matrix ***")
    print(confusion_matrix[dataset_list[0].name].keys())
    for task in task_branches:
        print(task.task_name, confusion_matrix[task.task_name].values())


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


def test_synthetic_samples_versus_normal(original_task_datasetInterface, added_task_datasetInterface, PATH_MODELS,record_keeper,
                                         device='cuda', new_classes_per_increment=1, number_increments=4, extra_new_cat_multi=1):
    print("***** Testing pseudo-rehearsal versus real samples")

    new_class_index = 26
    is_time_to_break = False

    is_save = True
    BATCH = 64
    EPOCH_IMPROVE_LIMIT = 20
    num_per_cat_gen_class_test = 1
    num_per_cat_gen_class_test = 1000

    EPOCH_CNN = 1
    EPOCH_CNN = 10
    BETA_CNN = (0.999, .999)
    LEARNR_CNN = 0.00025

    EPOCH_VAE = 1
    EPOCH_VAE = 100
    LAT_DIM_VAE = 50
    BETA_VAE = (0.5, 0.999)
    LEARNR_VAE = 0.00035

    task_DIS_orig_for_synthetic = copy.deepcopy(original_task_datasetInterface)
    task_DIS_New_for_synthetic = copy.deepcopy(added_task_datasetInterface)

    combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig_for_synthetic.name + " pseudo", task_DIS_orig_for_synthetic,
                                                     device, PATH_MODELS,record_keeper)

    for increment in range(0, number_increments):

        task_DIS_orig = copy.deepcopy(original_task_datasetInterface)
        #task_DIS_orig.reset_variables_to_initial_state()
        task_DIS_added = copy.deepcopy(added_task_datasetInterface)

        name = "increment" + str(increment) + "synth multi "+ str(extra_new_cat_multi)

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

            combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS+"VAE MNIST epochs200,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 89.18212963867188 v2",False)

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


            print("\n------------ Pseudo Samples", list_categories_to_add)

            print("\nPseudo Samples - create samples")
            combined_task_branch_synthetic.create_blended_dataset_with_synthetic_samples(task_DIS_New_for_synthetic,
                                                                                         list_categories_to_add_marginal,extra_new_cat_multi)

            combined_task_branch_no_synthetic = task.TaskBranch(task_DIS_orig.name + str(list_categories_to_add) +" real",
                                                                task_DIS_orig, device, PATH_MODELS,record_keeper)

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


            given_test_set_compare_synthetic_and_normal_approaches(combined_task_branch_synthetic, combined_task_branch_no_synthetic, task_DIS_orig,
                                                                   list_categories_to_add, extra_new_cat_multi)

            test_generating_and_classification_ability_multi_tasks([combined_task_branch_synthetic,combined_task_branch_no_synthetic], number_per_category=num_per_cat_gen_class_test, device=device)


            record_keeper.record_to_file("real_versus fake continual learning adding "+str(list_categories_to_add)+" synth multi "+str(extra_new_cat_multi))


def test_synthetic_samples_versus_normal_increasing(original_task_datasetInterface, PATH_MODELS,record_keeper,
                                         device='cuda',number_increments=10, extra_new_cat_multi=1):
    print("***** Testing pseudo-rehearsal versus real samples")

    new_class_index = 10
    is_time_to_break = False

    is_save = True
    BATCH = 64
    EPOCH_IMPROVE_LIMIT = 20
    num_per_cat_gen_class_test = 1000
    num_per_cat_gen_class_test = 1

    EPOCH_CNN = 1
    EPOCH_CNN = 10
    BETA_CNN = (0.999, .999)
    LEARNR_CNN = 0.00025

    EPOCH_VAE = 1
    EPOCH_VAE = 50
    LAT_DIM_VAE = 50
    BETA_VAE = (0.5, 0.999)
    LEARNR_VAE = 0.00035

    cat_list_all = original_task_datasetInterface.categories_list

    combined_task_branch_synthetic = None


    for increment in range(0, number_increments):

        task_DIS_orig = copy.deepcopy(original_task_datasetInterface)
        category_list_subset = cat_list_all[0:increment+1]
        task_DIS_orig.reduce_categories_in_dataset(category_list_subset)
        task_DIS_orig_for_synth = copy.deepcopy(task_DIS_orig)


        name = "increment" + str(increment) + "synth multi "+ str(extra_new_cat_multi)

        # combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig_for_synth.name + " pseudo",
        #                                                  task_DIS_orig_for_synth,
        #                                                  device, PATH_MODELS, record_keeper)
        # combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS+"VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 108.8118243938762 increment8synth multi 1",False)
        # combined_task_branch_synthetic.categories_list = category_list_subset[0:increment]
        # print("combined cat list",combined_task_branch_synthetic.categories_list)

        if (increment == 0):
            combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig_for_synth.name + " pseudo", task_DIS_orig_for_synth,
                                                             device, PATH_MODELS,record_keeper)

            print("Starting point, just MNIST....")
            print("Training VAE - Starting")

            print("\n Beginning point only ",category_list_subset)

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
            #
            # combined_task_branch_synthetic.create_and_train_CNN(model_id=name, num_epochs=EPOCH_CNN, batch_size=BATCH,
            #                                                     is_frozen=False,
            #                                                     is_off_shelf_model=True,
            #                                                     epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
            #                                                     learning_rate=LEARNR_CNN,
            #                                                     betas=BETA_CNN, is_save=is_save)

            combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS+"VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 101.1282752212213 increment0synth multi 1",False)
            combined_task_branch_synthetic.load_existing_CNN(PATH_MODELS+"CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment0synth multi 1")

            given_test_set_compare_synthetic_and_normal_approaches([combined_task_branch_synthetic], task_DIS_orig,category_list_subset, extra_new_cat_multi)

            #test_generating_and_classification_ability_multi_tasks([combined_task_branch_synthetic], number_per_category=num_per_cat_gen_class_test, device=device)



            record_keeper.record_to_file("real_versus fake continual learning adding "+str(category_list_subset[0])+" to "+str(category_list_subset[-1])+" synth multi "+str(extra_new_cat_multi))

            del task_DIS_orig
            torch.cuda.empty_cache()

        else:



            print("\n------------ Real Samples", category_list_subset)
            combined_task_branch_no_synthetic = task.TaskBranch(task_DIS_orig.name +"real ["+str(category_list_subset[0])+" to "+str(category_list_subset[-1])+ "] ",task_DIS_orig, device, PATH_MODELS, record_keeper)

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
            # combined_task_branch_no_synthetic.load_existing_VAE(PATH_MODELS+"VAE MNIST0 - zero to 9 - nine real epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 88.93284345703125 increment9synth multi 1",False)
            # combined_task_branch_no_synthetic.load_existing_CNN(PATH_MODELS+"CNN MNIST0 - zero to 9 - nine real epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9952000000000001 increment9synth multi 1")

            print("\n------------ Pseudo Samples ",category_list_subset)



            print("\nPseudo Samples - create samples")
            combined_task_branch_synthetic.create_blended_dataset_with_synthetic_samples(task_DIS_orig,[category_list_subset[-1]],extra_new_cat_multi)



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



            given_test_set_compare_synthetic_and_normal_approaches([combined_task_branch_synthetic, combined_task_branch_no_synthetic], task_DIS_orig,
                                                                   category_list_subset, extra_new_cat_multi)

            #test_generating_and_classification_ability_multi_tasks([combined_task_branch_synthetic,combined_task_branch_no_synthetic], number_per_category=num_per_cat_gen_class_test, device=device)


            record_keeper.record_to_file("real_versus fake continual learning adding "+str(category_list_subset)+" synth multi "+str(extra_new_cat_multi))

            del task_DIS_orig, combined_task_branch_no_synthetic
            torch.cuda.empty_cache()


def test_synthetic_samples_versus_normal_increasing_PRETRAINED_VAE(original_task_datasetInterface, PATH_MODELS,record_keeper,
                                         device='cuda',number_increments=10, extra_new_cat_multi=1):
    print("***** Testing pseudo-rehearsal versus real samples")

    new_class_index = 10
    is_time_to_break = False

    is_save = False
    BATCH = 64
    EPOCH_IMPROVE_LIMIT = 20
    num_per_cat_gen_class_test = 1000
    num_per_cat_gen_class_test = 1

    EPOCH_CNN = 10
    EPOCH_CNN = 1
    BETA_CNN = (0.999, .999)
    LEARNR_CNN = 0.00025

    EPOCH_VAE = 1
    EPOCH_VAE = 50
    LAT_DIM_VAE = 50
    BETA_VAE = (0.5, 0.999)
    LEARNR_VAE = 0.00035

    cat_list_all = original_task_datasetInterface.categories_list

    combined_task_branch_synthetic = None

    FOLDER = "real_versus_synth_models/"
    model_string = []
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 101.1282752212213 increment0synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 78.45619751514317 increment1synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 89.60624052141372 increment2synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 95.91765151826462 increment3synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 98.84272441048243 increment4synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 102.96877034505208 increment5synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 104.12568975369538 increment6synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 104.48726003921449 increment7synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 108.8118243938762 increment8synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 109.9276000371655 increment9synth multi 1")


    for increment in range(0, number_increments):

        task_DIS_orig = copy.deepcopy(original_task_datasetInterface)
        category_list_subset = cat_list_all[0:increment+1]
        task_DIS_orig.reduce_categories_in_dataset(category_list_subset)
        task_DIS_orig_for_synth = copy.deepcopy(task_DIS_orig)


        name = "increment" + str(increment) + "synth multi "+ str(extra_new_cat_multi)

        if (increment == 0):
            combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig_for_synth.name + " pseudo", task_DIS_orig_for_synth,
                                                             device, PATH_MODELS,record_keeper)

            print("Starting point, just MNIST....")
            print("Training VAE - Starting")

            print("\n Beginning point only ",category_list_subset)


            combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS+FOLDER+"VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 101.1282752212213 increment0synth multi 1",False)
            combined_task_branch_synthetic.load_existing_CNN(PATH_MODELS+FOLDER+"CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment0synth multi 1")

            given_test_set_compare_synthetic_and_normal_approaches([combined_task_branch_synthetic], task_DIS_orig,category_list_subset, extra_new_cat_multi)


            record_keeper.record_to_file("real_versus fake continual learning adding "+str(category_list_subset[0])+" to "+str(category_list_subset[-1])+" synth multi "+str(extra_new_cat_multi))

            del task_DIS_orig
            torch.cuda.empty_cache()

        else:


            print("\n------------ Pseudo Samples ",category_list_subset)



            print("\nPseudo Samples - create samples")
            combined_task_branch_synthetic.create_blended_dataset_with_synthetic_samples(task_DIS_orig,[category_list_subset[-1]],extra_new_cat_multi)



            print("\nPseudo Samples - Loading VAE")
            combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS + FOLDER + model_string[increment])
            combined_task_branch_synthetic.num_categories_in_task = increment+1


            print("\nPseudo Samples - CNN")
            combined_task_branch_synthetic.create_and_train_CNN(model_id=name, num_epochs=EPOCH_CNN, batch_size=BATCH,
                                                                is_frozen=False,
                                                                is_off_shelf_model=True,
                                                                epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                learning_rate=LEARNR_CNN,
                                                                betas=BETA_CNN, is_save=is_save)



            given_test_set_compare_synthetic_and_normal_approaches([combined_task_branch_synthetic], task_DIS_orig,category_list_subset, extra_new_cat_multi)


            record_keeper.record_to_file("real_versus fake continual learning adding "+str(category_list_subset)+" synth multi "+str(extra_new_cat_multi))

            del task_DIS_orig
            torch.cuda.empty_cache()


def given_test_set_compare_synthetic_and_normal_approaches(task_branch_list, dataSetInter,new_cats_added,extra_new_cat_multi):

    device = 'cuda'
    split_to_measure_against = 'val'

    for task_branch in task_branch_list:

        task_branch.run_end_of_training_benchmarks("real_versus fake continual learning adding "+str(new_cats_added)+" synth multi "+str(extra_new_cat_multi),dataSetInter)



        # data_loader_CNN = dataSetInter.return_data_loaders('CNN', BATCH_SIZE = 1)
        # data_loader_VAE = dataSetInter.return_data_loaders('VAE', BATCH_SIZE = 1)
        #
        #
        # task_branch.VAE_most_recent.eval()
        # task_branch.CNN_most_recent.eval()
        # task_branch.VAE_most_recent.to(device)
        # task_branch.CNN_most_recent.to(device)
        #
        # # To update record for mean and std deviation distance. This deletes old entries before calculating
        # by_category_record_of_recon_error_and_accuracy = {i: [0,0,0] for i in range(0, task_branch.num_categories_in_task)}
        #
        #
        # for i, (x, y) in enumerate(data_loader_VAE[split_to_measure_against]):
        #     x = x.to(device)
        #     y_original = y.item()
        #     y = idx2onehot(y.view(-1, 1), task_branch.num_categories_in_task)
        #     y = y.to(device)
        #
        #     with torch.no_grad():
        #         reconstructed_x, z_mu, z_var = task_branch.VAE_most_recent(x, y)
        #
        #     # loss
        #     loss = task_branch.VAE_most_recent.loss(x, reconstructed_x, z_mu, z_var)
        #
        #     by_category_record_of_recon_error_and_accuracy[y_original][0] += 1
        #     by_category_record_of_recon_error_and_accuracy[y_original][1] += loss.item()
        #
        #
        # for i, (x, y) in enumerate(data_loader_CNN[split_to_measure_against]):
        #     # reshape the data into [batch_size, 784]
        #     # print(x.size())
        #     # x = x.view(batch_size, 1, 28, 28)
        #     x = x.to(device)
        #     y_original = y.item()
        #     y = y.to(device)
        #
        #     with torch.no_grad():
        #         outputs = task_branch.CNN_most_recent(x)
        #         _, preds = torch.max(outputs, 1)
        #         # print(preds)
        #         #criterion = nn.CrossEntropyLoss()
        #         #loss = criterion(outputs.to(device), y)
        #
        #         correct = torch.sum(preds == y.data)
        #
        #     by_category_record_of_recon_error_and_accuracy[y_original][2] += correct.item()
        #
        # task_branch.VAE_most_recent.cpu()
        # task_branch.CNN_most_recent.cpu()
        #
        #
        # total_count =0
        # total_recon =0
        # total_correct =0
        # for category in by_category_record_of_recon_error_and_accuracy:
        #
        #     count = by_category_record_of_recon_error_and_accuracy[category][0]
        #     recon_ave = by_category_record_of_recon_error_and_accuracy[category][1]/count
        #     accuracy = by_category_record_of_recon_error_and_accuracy[category][2] /count
        #
        #     total_count += by_category_record_of_recon_error_and_accuracy[category][0]
        #     total_recon += by_category_record_of_recon_error_and_accuracy[category][1]
        #     total_correct += by_category_record_of_recon_error_and_accuracy[category][2]
        #
        #     print("For:",count, category," Ave. Recon:",recon_ave," Ave. Accuracy:",accuracy)
        #
        # print("For all (",total_count,"): Ave. Recon:",total_recon/total_count," Ave. Accuracy:",total_correct/total_count)


def compare_pretrained_task_branches(original_task_datasetInterface, added_task_datasetInterface, PATH_MODELS,record_keeper):


    device = 'cuda'
    task_DIS_orig = copy.deepcopy(original_task_datasetInterface)
    task_DIS_added = copy.deepcopy(added_task_datasetInterface)
    task_DIS_orig.add_outside_data_to_data_set(task_DIS_added, ['z'])
    combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig.name + " pseudo", task_DIS_orig,
                                                     device, PATH_MODELS,record_keeper)

    combined_task_branch_no_synthetic = task.TaskBranch(task_DIS_orig.name + str(['z']) + " real",
                                                        task_DIS_orig, device, PATH_MODELS, record_keeper)


    synth_tuple_z = ("VAE MNIST pseudo epochs100,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 101.06068173495206 increment1synth multi 1",
    "CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.992840909090909 increment1synth multi 1")
    real_tuple_z = ("VAE MNIST['z'] real epochs100,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 92.68595929181134 increment1synth multi 1",
    "CNN MNIST['z'] real epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9955555555555555 increment1synth multi 1")

    combined_task_branch_synthetic.load_existing_VAE(str(PATH_MODELS + str(synth_tuple_z[0])), False)
    combined_task_branch_synthetic.load_existing_CNN(str(PATH_MODELS + str(synth_tuple_z[1])))

    combined_task_branch_no_synthetic.load_existing_VAE(str(PATH_MODELS + str(real_tuple_z[0])),False)
    combined_task_branch_no_synthetic.load_existing_CNN(str(PATH_MODELS + str(real_tuple_z[1])))

    combined_task_branch_synthetic.generate_samples_to_display()
    combined_task_branch_no_synthetic.generate_samples_to_display()

    given_test_set_compare_synthetic_and_normal_approaches(combined_task_branch_synthetic,combined_task_branch_no_synthetic, task_DIS_orig,['z'], 1)

    record_keeper.record_to_file("real_versus fake continual learning adding " + str(['z']) + " synth multi " + str(1))


########################

    task_DIS_orig = copy.deepcopy(original_task_datasetInterface)
    task_DIS_added = copy.deepcopy(added_task_datasetInterface)
    task_DIS_orig.add_outside_data_to_data_set(task_DIS_added, ['z','y'])
    combined_task_branch_synthetic = task.TaskBranch(task_DIS_orig.name + " pseudo", task_DIS_orig,
                                                     device, PATH_MODELS, record_keeper)

    combined_task_branch_no_synthetic = task.TaskBranch(task_DIS_orig.name + str(['z', 'y']) + " real",
                                                        task_DIS_orig, device, PATH_MODELS, record_keeper)

    synth_tuple_z_y = ("VAE MNIST pseudo epochs100,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 107.99619313557943 increment2synth multi 1",
    "CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9989583333333334 increment2synth multi 1")
    real_tuple_z_y = ("VAE MNIST['z', 'y'] real epochs100,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 94.45959071718413 increment2synth multi 1",
    "CNN MNIST['z', 'y'] real epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 0.9941379310344827 increment2synth multi 1")

    combined_task_branch_synthetic.load_existing_VAE(PATH_MODELS + str(synth_tuple_z_y[0]), False)
    combined_task_branch_synthetic.load_existing_CNN(PATH_MODELS + str(synth_tuple_z_y[1]))

    combined_task_branch_no_synthetic.load_existing_VAE(PATH_MODELS + str(real_tuple_z_y[0]), False)
    combined_task_branch_no_synthetic.load_existing_CNN(PATH_MODELS + str(real_tuple_z_y[1]))

    combined_task_branch_synthetic.generate_samples_to_display()
    combined_task_branch_no_synthetic.generate_samples_to_display()

    given_test_set_compare_synthetic_and_normal_approaches(combined_task_branch_synthetic,combined_task_branch_no_synthetic, task_DIS_orig,['z','y'], 1)

    record_keeper.record_to_file("real_versus fake continual learning adding " + str(['z', 'y']) + " synth multi " + str(1))

def test_concept_drift_for_single_task(task_branch, shear_degree_max,shear_degree_increments, split, num_samples_to_check=100):

        print("*** Testing how task relativity changes with concept drift: ", task_branch.task_name)

        shear_degree_increment_num  = round(shear_degree_max/shear_degree_increments)
        original_transforms = task_branch.dataset_interface.transforms

        task_branch_no_recalibration_changes = copy.deepcopy(task_branch)
        task_branch_no_recalibration_changes.name = "No recalibration changes"

        is_save = True
        BATCH = 64
        EPOCH_IMPROVE_LIMIT = 20

        EPOCH_CNN = 3
        EPOCH_CNN = 1
        BETA_CNN = (0.999, .999)
        LEARNR_CNN = 0.00025

        EPOCH_VAE = 10
        EPOCH_VAE = 1
        LAT_DIM_VAE = 50
        BETA_VAE = (0.5, 0.999)
        LEARNR_VAE = 0.00035

        for increment in range(0,shear_degree_increment_num+1):

            shear_degree = increment*shear_degree_increments

            shear_trans = transforms.Compose([transforms.ToPILImage(),lambda img: transforms.functional.affine(img, angle=0, translate=(0, 0),scale=1, shear=shear_degree),transforms.ToTensor()])

            task_branch.dataset_interface.transforms = transforms.Compose([original_transforms,shear_trans])
            dataloader = task_branch.dataset_interface.return_data_loaders(branch_component='VAE', BATCH_SIZE=1)[split]

            reconstruction_error_average, task_relatedness = task_branch.given_task_data_set_find_task_relatedness(
                dataloader, num_samples_to_check=num_samples_to_check, shear_degree=shear_degree)

            print("   --- With shear degree:", shear_degree)
            print("   --- Reconstruction error average", reconstruction_error_average, " Task relatedness",task_relatedness, "Number of samples:", num_samples_to_check)

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
                                                                        learning_rate=LEARNR_VAE, betas=BETA_VAE,
                                                                        is_save=is_save)
                print("\nPseudo Samples - CNN")
                task_branch.create_and_train_CNN(model_id="recalibrated wth shear "+str(shear_degree), num_epochs=EPOCH_CNN,
                                                                        batch_size=BATCH,
                                                                        is_frozen=False,
                                                                        is_off_shelf_model=True,
                                                                        epoch_improvement_limit=EPOCH_IMPROVE_LIMIT,
                                                                        learning_rate=LEARNR_CNN,
                                                                        betas=BETA_CNN, is_save=is_save)

                task_branch.run_end_of_training_benchmarks("recalibration vs no recalibration", task_branch.dataset_interface)

            task_branch_no_recalibration_changes.run_end_of_training_benchmarks("recalibration vs no recalibration", task_branch.dataset_interface)



def load_VAE_models_and_display_syn_images(PATH_MODELS, task_branch):

    task_branch.VAE_most_recent = None
    task_branch.CNN_most_recent = None


    FOLDER = "real_versus_synth_models/"
    model_string = []
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 101.1282752212213 increment0synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 78.45619751514317 increment1synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 89.60624052141372 increment2synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 95.91765151826462 increment3synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 98.84272441048243 increment4synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 102.96877034505208 increment5synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 104.12568975369538 increment6synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 104.48726003921449 increment7synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 108.8118243938762 increment8synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 109.9276000371655 increment9synth multi 1")

    print("Synthetic multiplier x1.0")
    i=1
    for string in model_string:
        task_branch.load_existing_VAE(PATH_MODELS+FOLDER+string)
        task_branch.num_categories_in_task = i
        #print("num cats ",i,"load",string)
        #task_branch.generate_samples_to_display()
        i += 1

    task_branch.load_existing_CNN(PATH_MODELS+FOLDER+"CNN MNIST pseudo epochs10,batch64,pretrainedTrue,frozenFalse,lr0.00025,betas(0.999, 0.999) accuracy 1.0 increment9synth multi 1")
    task_branch.run_end_of_training_benchmarks("double check", is_save=False)

    model_string = []
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltFalse,lr0.00035,betas(0.5, 0.999)lowest_error 101.1282752212213 increment0synth multi 1")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 81.97313514563623 increment1synth multi 1.25")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 86.8226265794332 increment2synth multi 1.25")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 94.33633891675669 increment3synth multi 1.25")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 98.11629391048503 increment4synth multi 1.25")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 101.63875584279498 increment5synth multi 1.25")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 104.91211280493653 increment6synth multi 1.25")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 105.55814459472128 increment7synth multi 1.25")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 108.8795609685252 increment8synth multi 1.25")
    model_string.append("VAE MNIST pseudo epochs50,batch64,z_d50,synthFalse,rebuiltTrue,lr0.00035,betas(0.5, 0.999)lowest_error 110.63353334016243 increment9synth multi 1.25")

    print("Synthetic multiplier x0.8")
    i=1
    for string in model_string:
        task_branch.load_existing_VAE(PATH_MODELS+FOLDER+string)
        task_branch.num_categories_in_task = i
        #print("num cats ",i,"load",string)
        #task_branch.generate_samples_to_display()
        i += 1
