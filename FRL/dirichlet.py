import pickle
from collections import defaultdict
import os
import random
import numpy as np

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

def generate_signatures(labelPath, num_participants, alpha, sigPath):
    datasetSize = len(os.listdir(labelPath))
    labels = []
    for idx in range(1, 1+datasetSize):
        npdata = np.load(os.path.join(labelPath, f'label_{idx}.npy'))
        labels.append(int(npdata))

    trainerSamplesDict, _ = sample_dirichlet_train_data_train(labels, num_participants, alpha, force=False)

    for preparer, indices in trainerSamplesDict.items():
        for index in indices:
            path = os.path.join(sigPath, f'sig_{index}.npy')
            np.save(path, [preparer])
