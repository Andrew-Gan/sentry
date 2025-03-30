import pickle
from . import dirichlet
import os
import numpy as np
import pickle

def write_cifar10_data(path):
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

def write_cifar10_metadata(path, labels, num_participants, alpha):
    trainerSamplesDict, _ = dirichlet.sample_dirichlet(labels, num_participants, alpha, force=False)
    for preparer, indices in trainerSamplesDict.items():
        for index in indices:
            filepath = os.path.join(path, f'metadata/metadata_{index}.npy')
            np.save(filepath, [preparer, labels[index]])

def prepare_cifar10(path, num_participants, alpha):
    labels = write_cifar10_data(path)
    write_cifar10_metadata(path, labels, num_participants, alpha)
    
