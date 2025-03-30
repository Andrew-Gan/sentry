from datasets import load_dataset
import numpy as np
import os
from . import dirichlet

def write_medicalqa_data(path):
    index = 0
    labels = []
    dataPath = os.path.join(path, 'data')
    os.makedirs(dataPath, exist_ok=True)
    queries = load_dataset('lavita/medical-qa-shared-task-v1-toy')
    
    labels = []
    for index, query in enumerate(queries):
        arr = np.frombuffer(query['startphrase'], dtype=np.uint8)
        np.save(os.path.join(path, f'data/data_{index}.npy'), arr)
        labels.append(query['label'])
        index += 1

    return labels

def write_medicalqa_metadata(path, labels, num_participants, alpha):
    trainerSamplesDict, _ = dirichlet.sample_dirichlet(labels, num_participants, alpha, force=False)
    for preparer, indices in trainerSamplesDict.items():
        for index in indices:
            filepath = os.path.join(path, f'metadata/metadata_{index}.npy')
            np.save(filepath, [preparer, labels[index]])

def prepare_medicalqa(path, num_participants=1, alpha=1):
    labels = write_medicalqa_data(path)
    write_medicalqa_metadata(path, labels, num_participants, alpha)
