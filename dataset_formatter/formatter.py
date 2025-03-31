from datasets import load_dataset
import numpy as np
import os
from . import dirichlet

def write_cifar10_data(path, samples, num_participants, alpha):
    labels = []
    for index, sample in enumerate(samples):
        image = np.array(sample['img'])
        data = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))
        np.save(os.path.join(path, f'data/data_{index}.npy'), data)
        labels.append(sample['label'])
    
    os.makedirs(os.path.join(path, 'metadata'), exist_ok=True)
    trainerSamplesDict, _ = dirichlet.sample_dirichlet(labels, num_participants, alpha, force=False)
    for preparer, indices in trainerSamplesDict.items():
        for index in indices:
            filepath = os.path.join(path, f'metadata/metadata_{index}.npy')
            np.save(filepath, [preparer, labels[index]])

def write_hellaswag_data(path, samples):
    os.makedirs(os.path.join(path, 'metadata'), exist_ok=True)
    for index, query in enumerate(samples):
        arr = np.frombuffer(query['ctx'].encode('utf-8'), dtype=np.uint8)
        np.save(os.path.join(path, f'data/data_{index}.npy'), arr)

        arr = np.frombuffer(query['activity_label'].encode('utf-8'), dtype=np.uint8)
        np.save(os.path.join(path, f'metadata/metadata_{index}.npy'), [0, 0])

def prepare_data(path, dataset_name, num_participants, alpha):
    os.makedirs(os.path.join(path, 'data'), exist_ok=True)
    samples = load_dataset(dataset_name)['train']
    if dataset_name == 'uoft-cs/cifar10':
        write_cifar10_data(path, samples, num_participants, alpha)
    elif dataset_name == 'Rowan/hellaswag':
        write_hellaswag_data(path, samples)
    else:
        raise NotImplementedError('Dataset processing not implemented')
