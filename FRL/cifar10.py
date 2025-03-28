import pickle
from collections import defaultdict
import os
import random
import numpy as np
import pickle

def sample_dirichlet_train_data_train(train_dataset, num_participants, alpha, force=False):
    tr_classes = {}

    for idx, label in enumerate(train_dataset):
        if label in tr_classes:
            tr_classes[label].append(idx)
        else:
            tr_classes[label] = [idx]

    tr_per_participant_list = defaultdict(list)
    tr_per_participant_list_labels_fr = defaultdict(defaultdict)

    tr_no_classes = len(tr_classes.keys())

    for n in range(tr_no_classes):
        random.shuffle(tr_classes[n])

        tr_class_size=len(tr_classes[n])
        d_sample = np.random.dirichlet(np.array(num_participants * [alpha]))
        tr_sampled_probabilities = tr_class_size * d_sample ##prob of selecting for this class
        for user in range(num_participants):
            no_imgs = int(round(tr_sampled_probabilities[user]))
            sampled_list = tr_classes[n][:min(len(tr_classes[n]), no_imgs)]
            random.shuffle(sampled_list)
            tr_per_participant_list_labels_fr[user][n]=len(sampled_list)
            tr_per_participant_list[user].extend(sampled_list[:])
            tr_classes[n] = tr_classes[n][min(len(tr_classes[n]), no_imgs):]

    return tr_per_participant_list, tr_per_participant_list_labels_fr

def process_cifar10_data(path):
    index = 0
    labels = []
    dataPath = os.path.join(path, 'data')
    os.makedirs(dataPath, exist_ok=True)

    for i in range(1, 6):
        train_file = os.path.join(path, 'cifar-10-batches-py', 'data_batch_' + f'{i}')
        with open(train_file, 'rb') as f:
            train_dict = pickle.load(f, encoding='bytes')
            for data, label in zip(train_dict[b'data'], train_dict[b'labels']):
                data = np.transpose(np.reshape(data, (3, 32, 32)), (1, 2, 0))
                np.save(os.path.join(path, f'data/data_{index}.npy'), data)
                labels.append(label)
                index += 1

    return labels

def process_cifar10_metadata(path, labels, num_participants, alpha):
    metaPath = os.path.join(path, 'metadata')
    os.makedirs(metaPath, exist_ok=True)
    trainerSamplesDict, _ = sample_dirichlet_train_data_train(labels, num_participants, alpha, force=False)
    for preparer, indices in trainerSamplesDict.items():
            for index in indices:
                path = os.path.join(metaPath, f'metadata_{index}.npy')
                np.save(path, [preparer, labels[index]])

def prepare_cifar10(path, num_participants, alpha):
    labels = process_cifar10_data(path)
    process_cifar10_metadata(path, labels, num_participants, alpha)
    
