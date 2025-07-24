from collections import defaultdict
import random
import numpy as np

def sample_dirichlet(train_dataset, num_participants, alpha, force=False):
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

def write_cifar10_data(path, samples, num_participants, alpha):
    labels = []
    for index, sample in enumerate(samples):
        image = np.array(sample['img'])
        data = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))
        np.save(os.path.join(path, f'data/data_{index}.npy'), data)
        labels.append(sample['label'])

    trainerSamplesDict, _ = sample_dirichlet(labels, num_participants, alpha, force=False)
    for preparer, indices in trainerSamplesDict.items():
        for index in indices:
            filepath = os.path.join(path, f'metadata/metadata_{index}.npy')
            np.save(filepath, [preparer, labels[index]])

def write_hellaswag_data(path, samples):
    for index, query in enumerate(samples):
        arr = np.frombuffer(query['ctx'].encode('utf-8'), dtype=np.uint8)
        np.save(os.path.join(path, f'data/data_{index}.npy'), arr)

        arr = np.frombuffer(query['activity_label'].encode('utf-8'), dtype=np.uint8)
        np.save(os.path.join(path, f'metadata/metadata_{index}.npy'), [0, 0])
