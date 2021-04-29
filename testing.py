import numpy as np
import pandas as pd
import math, copy
import re, random
from collections import defaultdict
from itertools import product
from itertools import chain
from collections import Counter
import time
import sys


def generate_graph(cliques_list, n):
    graph = np.zeros(shape=(n,n))
    for clique in cliques_list:
        if len(clique) != 1:
            for i in range(len(clique) - 1):
                    graph[int(clique[i])][int(clique[-1])] = 1
    return graph


def generate_tables(uai_network, cliques_list):
    """
    The fucntion generates a hash map of factors with its cliques as keys.
    """
    n = int(uai_network[1])
    size = len(uai_network)
    factor_table = defaultdict(list)
    cnt = 0
    prob_value = []
    for line in range(4 + n + 1, size):
        # print(uai_network[line])
        if uai_network[line].isnumeric():
            list_size = int(uai_network[line]) // 2
            values = []
            for i in range(1, list_size + 1):
                values.extend(uai_network[line+i].split(' '))
            values = list(map(float, values))
            table_vals = list(product((0,1), repeat=len(cliques_list[cnt])))
            factor_table[tuple(cliques_list[cnt])] = {'factors': np.array(values), 'truth_table':table_vals}
            cnt += 1
    return factor_table


def read_FOD(fod_filepath):
    f = open(fod_filepath,'r')
    train_file = f.readlines()
    train_file = list(map(lambda s: s.strip(), train_file))
    initial = train_file[0].split(' ')
    no_of_vars, no_of_samples = int(initial[0]), int(initial[1])
    trainin_data = []
    sample_dict = defaultdict(list)
    for sample in train_file[1:]:
        data = sample.split(' ')
        data = np.array(list(map(int, data)))
        trainin_data.append(data)
    trainin_data = np.array(trainin_data)
    for i in range(no_of_vars):
        sample_dict[i] = trainin_data[:,i]
    return sample_dict


def read_POD(pod_filepath):
    f = open(pod_filepath,'r')
    train_file = f.readlines()
    train_file = list(map(lambda s: s.strip(), train_file))
    initial = train_file[0].split(' ')
    no_of_vars, no_of_samples = int(initial[0]), int(initial[1])
    trainin_data = []
    sample_dict = defaultdict(list)
    for sample in train_file[1:]:
        data = sample.split(' ')
        for i in range(len(data)):
            if data[i] == '?':
                data[i] = -99
            else:
                data[i] = int(data[i])
        data = np.array(data)
        trainin_data.append(data)
    trainin_data = np.array(trainin_data)
    return trainin_data

def get_parents(cliques_list):
    parents_map = defaultdict(list)
    for clique in cliques_list:
        if clique[-1] not in parents_map:
            parents_map[clique[-1]] = clique[:-1]
    return parents_map

def get_observed_probability(truth_table, clique, sample_dict):
    no_of_sample = len(sample_dict[0])
    no_of_vals = len(truth_table)
    freq_vals = []
    child_node = clique[-1]
    for truth_val in truth_table:
        commons = list(range(no_of_sample))
        parent_cnt, child_cnt = 0, 0
        for i in range(len(clique)):
            idx = np.where(sample_dict[clique[i]] == truth_val[i])
            commons = np.intersect1d(commons, idx)
            if clique[i] == child_node:
                child_cnt = len(commons)
            else:
                parent_cnt = len(commons)
            # print(clique[i], truth_val[i], commons, cnts)
            probability = (1 + child_cnt) / (2 + parent_cnt)
            freq_vals.append(probability)
    return freq_vals


def get_LLDiff(factor_table, learned_parameters, test_samples):
    log_likelihood = 0
    for sample in test_samples:
        given, learned = 0, 0
        for clique in learned_parameters.keys():
            truth_val = np.take(sample, clique)
            idx = factor_table[clique]['truth_table'].index(tuple(truth_val))
            given += np.log10(factor_table[clique]['factors'][idx])
            learned += np.log10(learned_parameters[clique]['weights'][idx])
        log_likelihood += abs(given - learned)
    print("LLDiff is: {}".format(log_likelihood))
    return log_likelihood

def task_1(factor_table, fod_train_filepath, fod_test_filepath):
    sample_dict = read_FOD(fod_train_filepath)
    test_samples = read_POD(fod_test_filepath)
    learned_parameters = copy.deepcopy(factor_table)
    for clique in factor_table.keys():
        freq_vals = get_observed_probability(factor_table[clique]['truth_table'], clique, sample_dict)
        learned_parameters[clique]['weights'] = freq_vals
    LLDiff = get_LLDiff(factor_table, learned_parameters, test_samples)
    print("\n\n\n TASK 1 LLDiff: ", LLDiff)
    return LLDiff


def normalize_array(array):
    local_sum = np.sum(array)
    return (array / local_sum)


def generate_random_params(clique):
    n = 2**len(clique)
    random_prob = np.random.uniform(0,1,n)
    # for i in range(0,len(random_prob),2):
    #     random_prob[i + 1] = 1 - random_prob[i]
    return random_prob


def generate_sample(sample):
    missing_vars = list(np.where(sample == -99)[0])
    rep = 2**len(missing_vars)
    sample_combinations = [copy.deepcopy(sample) for _ in range(rep)]
    missing_combination = list(product((0,1), repeat=int(np.log2(rep))))
    missing_combination = list(map(list, missing_combination))
    for i in range(rep):
        sample_combinations[i][missing_vars] = missing_combination[i]
    return sample_combinations


def get_normalize_weights(weights):
    total_sum = sum(weights.values())
    for sample, probability in weights.items():
        weights[sample] = probability / total_sum
    return weights


def calculate_weights(EM_parameters, generated_samples):
    saved_weights = {}
    for sample in generated_samples:
        probability_weight = 1
        for clique, value in EM_parameters.items():
            truth_val = np.take(sample, clique)
            idx = EM_parameters[clique]['truth_table'].index(tuple(truth_val))
            probability_weight *= EM_parameters[clique]['weights'][idx]
        saved_weights[tuple(sample)] = probability_weight
    return saved_weights


def E_step(EM_parameters, samples):
    weights = {}
    for sample in samples:
        generated_samples = generate_sample(sample)
        saved_weights = calculate_weights(EM_parameters, generated_samples)
        normalized_weights = get_normalize_weights(saved_weights)
        weights.update(normalized_weights)
    return EM_parameters, weights

def M_step(EM_parameters, weights):
    for clique in EM_parameters.keys():
        for index, truth_val in enumerate(EM_parameters[clique]['truth_table']):
            _local_sum = 0
            matched_sum = 0
            for sample, probability_weight in weights.items():
                sample_truth = np.take(sample, clique)
                if (np.array(sample_truth, dtype=int) == np.array(truth_val, dtype=int)).all():
                    matched_sum += probability_weight
                if (np.array(sample_truth[:-1], dtype=int) == np.array(truth_val[:-1], dtype=int)).all():
                    _local_sum += probability_weight
            local_prob = (1 + matched_sum) / (2 + _local_sum)
            EM_parameters[clique]['weights'][index] = local_prob
        for _iter in range(0,len(EM_parameters[clique]['weights']),2):
            EM_parameters[clique]['weights'][_iter + 1] = 1 - EM_parameters[clique]['weights'][_iter]
        # EM_parameters[clique]['weights'] = normalize_array(EM_parameters[clique]['weights'])
    return EM_parameters


def EM_algorithm(factor_table, samples):
    EM_parameters = {}

    for clique in factor_table.keys():
        _weights = generate_random_params(clique)
        _truth_table = factor_table[clique]['truth_table']
        EM_parameters[clique] = {'weights': _weights, 'truth_table': _truth_table}

    for iter in range(20):
        EM_parameters, weights = E_step(EM_parameters, samples)
        EM_parameters = M_step(EM_parameters, weights)
    # print("After 1 iter ", EM_parameters)
    return EM_parameters

def task_2(factor_table, pod_train_filepath, test_samples):
    samples = read_POD(pod_train_filepath)
    EM_parameters = EM_algorithm(factor_table, samples)
    LLDiff = get_LLDiff(factor_table, EM_parameters, test_samples)
    return LLDiff

def main():
    dataset_source_path = '/Users/abhishekhosmani/CompSci/College/AdvStats/Assignments/hw4/data/dataset1'
    uai_path = dataset_source_path + '/1.uai'
    test_filepaths = dataset_source_path + '/test.txt'
    test_samples = read_POD(test_filepaths)
    f = open(uai_path,'r')
    uai_network = f.readlines()
    uai_network = list(map(lambda s: s.strip(), uai_network))
    n = int(uai_network[1])
    cardinality = uai_network[2].split(" ")
    no_of_cliques = int(uai_network[3])
    cliques_list = []
    for i in range(4,4 + no_of_cliques):
        _list = uai_network[i].split("\t")
        _list = list(map(int, _list))
        cliques_list.append(_list[1:])

    factor_table = generate_tables(uai_network, cliques_list)
    graph = generate_graph(cliques_list, n)
    
    # print(graph[92])
    parents_map = get_parents(cliques_list)
    fod_train_filepaths = ['train-f-1.txt', 'train-f-2.txt', 'train-f-3.txt','train-f-4.txt']
    pod_train_filepaths = ['train-p-1.txt', 'train-p-2.txt', 'train-p-3.txt','train-p-4.txt']
    seeds = random.sample(range(1, 100),5)
    task1_LLDiff = []
    task2_LLDiff = []

    # for train_filepath in fod_train_filepaths[:2]:
    #     train_file = dataset_source_path + '/' + train_filepath
    #     for k in range(5):
    #         np.random.seed(seeds[k])
    #         task1_LLDiff.append(task_1(factor_table, train_file, test_samples))
    #     print("TASK 1: training file {} has \n Mean :{} \n St.dev :{}".format(train_filepath, np.mean(task1_LLDiff), np.std(task1_LLDiff)))


    for train_filepath in pod_train_filepaths[:2]:
        train_file = dataset_source_path + '/' + train_filepath
        for k in range(5):
            np.random.seed(seeds[k])
            task2_LLDiff.append(task_2(factor_table, train_file, test_samples))
        print("TASK 2: training file {} has \n Mean :{} \n St.dev :{}".format(train_filepath, np.mean(task2_LLDiff), np.std(task2_LLDiff)))
    


    
main()

