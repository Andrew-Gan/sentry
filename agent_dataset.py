import dataset_formatter.formatter as formatter
import sys
import os
import shutil
from datasets import load_dataset

if __name__ == '__main__':
    try:
        path = sys.argv[4]
        dataset_name = sys.argv[1]
        num_sources = int(sys.argv[2])
        alpha = int(sys.argv[3])
    except:
        raise SyntaxError('Usage: python agent_dataset.py <identifier> <num_sources> <alpha> <path>')

    data_path = os.path.join(path, 'data')
    meta_path = os.path.join(path, 'metadata')
    shutil.rmtree(data_path, ignore_errors=True)
    shutil.rmtree(meta_path, ignore_errors=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(meta_path, exist_ok=True)

    samples = load_dataset(dataset_name)['train']
    if dataset_name == 'uoft-cs/cifar10':
        write_cifar10_data(path, samples, num_participants, alpha)
    elif dataset_name == 'Rowan/hellaswag':
        write_hellaswag_data(path, samples)
    else:
        raise NotImplementedError('Dataset processing not implemented')
