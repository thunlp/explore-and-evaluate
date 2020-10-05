import os, sys
import argparse
import torch
import random
import numpy as np
from sklearn.svm import SVC
from util import print_time_info, set_random_seed, get_hits, topk
from tqdm import tqdm


def sim_standardization(sim):
    mean = np.mean(sim)
    std = np.std(sim)
    sim = (sim - mean) / std
    return sim

def load_partial_sim(sim_path, standardization=True):
    partial_sim = np.load(sim_path)
    sim_matrix = partial_sim[0]
    sim_indice = partial_sim[1]
    sim_indice = sim_indice.astype(np.int)
    assert sim_matrix.shape == sim_indice.shape
    if standardization:
        sim_matrix = sim_standardization(sim_matrix)
    size = sim_matrix.shape[0]
    sim = np.zeros((size, size), dtype=np.float)
    np.put_along_axis(sim, sim_indice, sim_matrix, 1)
    return sim, sim_matrix.shape


def load_sim_matrices(data_set, model_name_list, load_hard=True):
    train_sims = []
    valid_sims = []
    test_sims = []
    data_set = data_set.split('/')[-1]

    for model_name in tqdm(model_name_list):
        if load_hard:
            train_sim_path = "./log/grid_search_hard_%s_%s/train_sim.npy" % (model_name, data_set)
            train_sim = np.load(train_sim_path)
            train_sims.append(train_sim)
            valid_sim_path = "./log/grid_search_hard_%s_%s/valid_sim.npy" % (model_name, data_set)
            valid_sim = np.load(valid_sim_path)
            valid_sims.append(valid_sim)
            test_sim_path = "./log/grid_search_hard_%s_%s/test_sim.npy" % (model_name, data_set)
            test_sim = np.load(test_sim_path)
            test_sims.append(test_sim)
        else:
            train_sim_path = "./log/grid_search_%s_%s/train_sim.npy" % (model_name, data_set)
            train_sim = np.load(train_sim_path)
            train_sims.append(train_sim)
            valid_sim_path = "./log/grid_search_%s_%s/valid_sim.npy" % (model_name, data_set)
            valid_sim = np.load(valid_sim_path)
            valid_sims.append(valid_sim)
            # if model_name == 'Name':
            #     print('I am loading here')
            #     test_sim_path = "./log/grid_search_%s3_%s/test_sim.npy" % (model_name, data_set)
            # else:
            test_sim_path = "./log/grid_search_%s_%s/test_sim.npy" % (model_name, data_set)
            test_sim = np.load(test_sim_path)
            test_sims.append(test_sim)
    return train_sims, valid_sims, test_sims


def generate_data(sims, ratio):
    assert sims[0].shape[0] == sims[0].shape[1]
    for i in range(1, len(sims)):
        assert sims[i].shape == sims[i - 1].shape
    sim_num = len(sims)
    size = sims[0].shape[0]
    sims = [np.reshape(sim, (size, size, 1)) for sim in sims]
    sims = np.concatenate(sims, axis=-1)  # shape = [size, size, sim_num]
    assert sims.shape == (size, size, sim_num)

    positive_data = [sims[i, i] for i in range(size)]
    negative_indice = np.random.randint(low=0, high=size, size=(ratio * size, 2))
    negative_indice = [(x, y) for x, y in negative_indice if x != y]

    negative_data = [sims[x, y] for x, y in negative_indice]
    data = positive_data + negative_data
    label = [1 for _ in range(len(positive_data))] + [0 for _ in range(len(negative_data))]
    data = [f.reshape(1, sim_num) for f in data]

    ## shuffle
    tmp_box = list(zip(data, label))
    random.shuffle(tmp_box)
    data, label = zip(*tmp_box)

    data = np.concatenate(data, axis=0)
    label = np.asarray(label)
    return data, label


def ensemble_sims_with_svm(train_sims, valid_sims, test_sims, device, avg=False):
    set_random_seed()

    def sim_standardization2(sim):
        mean = np.mean(sim)
        std = np.std(sim)
        sim = (sim - mean) / std
        return sim, mean, std

    def sim_standardization3(sim, mean, std):
        return (sim - mean) / std

    train_sims2 = []
    mean_list = []
    std_list = []
    for sim in train_sims:
        sim, mean, std = sim_standardization2(sim)
        train_sims2.append(sim)
        mean_list.append(mean)
        std_list.append(std)

    train_sims = train_sims2
    valid_sims = [sim_standardization3(sim, mean_list[i], std_list[i]) for i, sim in enumerate(valid_sims)]
    test_sims = [sim_standardization3(sim, mean_list[i], std_list[i]) for i, sim in enumerate(test_sims)]

    if avg:
        get_hits(sum(test_sims), device=device)
        return

    train_data, train_label = generate_data(train_sims, ratio=len(test_sims) * 4)
    test_data, test_label = generate_data(test_sims, ratio=1)

    def ensemble_sims_with_weight(test_sims, weight):
        ## test performance
        test_size = test_sims[0].shape[0]
        test_sims = [sim.reshape(test_size, test_size, 1) for sim in test_sims]
        test_sims = np.concatenate(test_sims, axis=-1)
        test_sims = np.dot(test_sims, weight)
        test_sims = np.squeeze(test_sims, axis=-1)
        return - test_sims

    def performance_svc(train_data, train_label, test_sims, C):
        clf = SVC(kernel='linear', C=C, gamma='auto')
        clf.fit(train_data, train_label)
        prediction = clf.predict(test_data)
        print_time_info('Classification accuracy: %f.' % (np.sum(prediction == test_label) / len(test_label)))
        weight = clf.coef_.reshape(-1, 1)  # shape = [sim_num, 1]
        test_sims = ensemble_sims_with_weight(test_sims, weight)
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(test_sims, print_info=False, device=device)
        top1 = (top_lr[0] + top_rl[0]) / 2
        return top1, weight

    C_range = [1e-6, 1e-5] #1e-4, 1e-3, 1e-2, 1e-1]# 1, 10, 100, 1000]
    best_top1 = 0
    best_C = 0
    best_weight = None
    for C in C_range:
        top1, weight = performance_svc(train_data, train_label, valid_sims, C)
        if top1 > best_top1:
            best_top1 = top1
            best_C = C
            best_weight = weight
    test_sims = ensemble_sims_with_weight(test_sims, best_weight)
    print('Best C=%f.' % best_C)
    print('Weight', best_weight.reshape(-1))
    get_hits(test_sims, device=device)


def ensemble_partial_sim_matrix(data_set, svm=False, device='cpu'):
    def partial_get_hits(sim, top_k=(1, 10), kg='source', print_info=True):
        if isinstance(sim, np.ndarray):
            sim = torch.from_numpy(sim)
        top_lr, mr_lr, mrr_lr = topk(sim, top_k, device=device)
        if print_info:
            print_time_info('For each %s:' % kg, dash_top=True)
            print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_lr, mrr_lr))
            for i in range(len(top_lr)):
                print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_lr[i]))
        return top_lr, mr_lr, mrr_lr

    def load_partial_sim_list(sim_path_list):
        sim = None
        shape = None
        for sim_path in tqdm(sim_path_list):
            target, sim_matrix_shape = load_partial_sim(sim_path)
            if shape == None:
                shape = sim_matrix_shape
            else:
                assert shape == sim_matrix_shape
            if sim is not None:
                assert sim.shape == target.shape
                sim = sim + target
            else:
                sim = target
        sim = sim / len(sim_path_list)
        return sim

    data_set = data_set.split('DWY100k/')[1]

    # init sim_list
    model_name_list = ['Literal', 'Structure', 'Digital', "Name"]
    sim_path_list = ["./log/grid_search_%s_%s/test_sim.npy" % (model, data_set) for model in model_name_list]
    sim_t_path_list = ["./log/grid_search_%s_%s/test_sim_t.npy" % (model, data_set) for model in model_name_list]
    if not svm:
        partial_get_hits(load_partial_sim_list(sim_path_list), kg='source')
        partial_get_hits(load_partial_sim_list(sim_t_path_list), kg='target')
        print_time_info('-------------------------------------')
        return

    def svm_ensemble(train_sim_path_list, valid_sim_path_list, test_sim_path_list, T=False):
        positive_data = []  # shape = [sim_num, size]
        negative_data = []  # shape = [sim_num, size * ratio]
        sim_num = len(train_sim_path_list)

        size = 30000
        negative_indice = np.random.randint(low=0, high=size, size=(4 * sim_num * size, 2))
        negative_indice = [(x, y) for x, y in negative_indice if x != y]
        for sim_path in tqdm(train_sim_path_list, desc='Load train sims'):
            sim, _ = load_partial_sim(sim_path)
            assert size == sim.shape[0]
            positive_data.append([sim[i, i] for i in range(size)])
            negative_data.append([sim[x, y] for x, y in negative_indice])

        positive_data = np.asarray(positive_data).T  # shape = [size, sim_num]
        negative_data = np.asarray(negative_data).T  # shape = [size * ratio, sim_num]
        print(positive_data.shape)
        print(negative_data.shape)

        valid_sims = []
        for sim_path in tqdm(valid_sim_path_list, desc='Load valid sims'):
            sim = np.load(sim_path)
            if T:
                sim = sim.T
            valid_sims.append(np.expand_dims(sim, -1))
        valid_sims = np.concatenate(valid_sims, axis=-1)  # shape = [size, size, sim_num]

        data = np.concatenate([positive_data, negative_data], axis=0)
        label = [1 for _ in range(len(positive_data))] + [0 for _ in range(len(negative_data))]
        label = np.asarray(label)

        C_range = [1e-6, 1e-5] #[1e-1, 1, 10, 1000]
        best_C = 0
        best_top1 = 0
        best_weight = None
        for C in tqdm(C_range, desc='Fitting SVM'):
            clf = SVC(kernel='linear', C=C, gamma='auto')
            clf.fit(data, label)
            weight = clf.coef_.reshape(-1, 1)
            tmp_valid_sims = np.dot(valid_sims, weight)
            tmp_valid_sims = np.squeeze(tmp_valid_sims, axis=-1)
            top_lr, mr_lr, mrr_lr = partial_get_hits(-tmp_valid_sims, print_info=False)
            top1 = top_lr[0]
            if top1 > best_top1:
                best_top1 = top1
                best_weight = weight
                best_C = C
        print('Best C=%f' % best_C)
        print('Best weight', best_weight.reshape(-1))
        target_sim = None
        for idx, sim_path in tqdm(enumerate(test_sim_path_list), desc='Testing'):
            if target_sim is None:
                target_sim = best_weight[idx][0] * load_partial_sim(sim_path)[0]
            else:
                target_sim += best_weight[idx][0] * load_partial_sim(sim_path)[0]
        kg = 'source' if not T else 'target'
        partial_get_hits(-target_sim, kg=kg)

    train_sim_path_list = ["./log/grid_search_%s_%s/train_sim.npy" % (model, data_set) for model in model_name_list]
    train_sim_t_path_list = ["./log/grid_search_%s_%s/train_sim_t.npy" % (model, data_set) for model in model_name_list]
    valid_sim_path_list = ["./log/grid_search_%s_%s/valid_sim.npy" % (model, data_set) for model in model_name_list]
    test_sim_path_list = ["./log/grid_search_%s_%s/test_sim.npy" % (model, data_set) for model in model_name_list]
    test_sim_t_path_list = ["./log/grid_search_%s_%s/test_sim_t.npy" % (model, data_set) for model in model_name_list]
    svm_ensemble(train_sim_path_list, valid_sim_path_list, test_sim_path_list, T=False)
    svm_ensemble(train_sim_t_path_list, valid_sim_path_list, test_sim_t_path_list, T=True)


def load_bert_sim(dataset, load_hard_split):
    from load_data import _load_seeds
    from pathlib import Path
    def matrix_sample(sim, indices):
        # indices should be in increasing order
        sim = sim[indices]
        sim = sim[:, indices]
        return sim

    bert_sim_path = './bin/%s/running_temp/bert_base_sim_matrix.npy' % dataset
    bert_sim = np.load(bert_sim_path)
    train_entity_seeds, valid_entity_seeds, test_entity_seeds, entity_seeds = _load_seeds(Path('./bin/%s/' % dataset),
                                                                                          0.3, load_hard_split)
    train_sr_id_set = {seed[0] for seed in train_entity_seeds}
    train_indice = np.asarray([idx for idx, seed in enumerate(entity_seeds) if seed[0] in train_sr_id_set])
    bert_sim_train = matrix_sample(bert_sim, train_indice)

    valid_sr_id_set = {seed[0] for seed in valid_entity_seeds}
    valid_indice = np.asarray([idx for idx, seed in enumerate(entity_seeds) if seed[0] in valid_sr_id_set])
    bert_sim_valid = matrix_sample(bert_sim, valid_indice)

    test_sr_id_set = {seed[0] for seed in test_entity_seeds}
    test_indice = np.asarray([idx for idx, seed in enumerate(entity_seeds) if seed[0] in test_sr_id_set])
    bert_sim_test = matrix_sample(bert_sim, test_indice)
    return bert_sim_train, bert_sim_valid, bert_sim_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--svm', action='store_true')
    parser.add_argument('--load_hard_split', action='store_true')
    args = parser.parse_args()
    device = 'cuda:%d' % args.gpu_id if args.gpu_id >= 0 else 'cpu'

    # bert_sim_train, bert_sim_valid, bert_sim_test = load_bert_sim(args.dataset, args.load_hard_split)

    if args.dataset.find('DBP15k') >= 0:
        train_sims, valid_sims, test_sims = load_sim_matrices(args.dataset, ['Structure', 'Literal', 'Digital', 'Name'], args.load_hard_split)
        
        if not args.svm:
            # '''Ensemble with average pooling'''
            # test_sims.append(bert_sim_test)
            # test_sims = [sim_standardization(sim) for sim in test_sims]
            # get_hits(sum(test_sims), device=device)
            ensemble_sims_with_svm(train_sims, valid_sims, test_sims, device=device, avg=True)
        else:
            '''Ensemble with svm'''
            ensemble_sims_with_svm(train_sims, valid_sims, test_sims, device=device)
    elif args.dataset.find('DWY100k') >= 0:
        ensemble_partial_sim_matrix(args.dataset, svm=args.svm, device=device)