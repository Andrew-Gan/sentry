from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import os
import time
import numpy as np
import cupy as cp

from src.sign import build_hasher, sign, HashType, Topology, InputType

# To run with different data, see documentation of nvidia.dali.fn.readers.file
# points to https://github.com/NVIDIA/DALI_extra
data_file = os.path.join('cifar-10-batches-py', 'data')
labels_file = os.path.join('cifar-10-batches-py', 'labels')

def loss_func(pred, y):
    pass


def model(x):
    pass


def backward(loss, model):
    pass

hasher = build_hasher(HashType.LATTICE, Topology.ADD)

def lattice_hash(x):
    total = cp.empty(shape=((len(x),) + x[0].shape), dtype=np.uint8)
    for i, v in enumerate(x):
        total[i] = v
    x = hasher.update({'dataset': total}, blockSize=x[0].nbytes)

@pipeline_def(num_threads=8, device_id=0)
def get_dali_pipeline(device='cpu', gpu_direct=False, sign_dataset=False):
    # read from file system
    if gpu_direct:
        if device=='cpu':
            raise RuntimeError('Cannot use DALI on CPU with GPUDirect')
        images = fn.readers.numpy(file_root=data_file, random_shuffle=True,
            device='gpu', name='Reader', seed=0)
        labels = fn.readers.numpy(file_root=labels_file, random_shuffle=True,
            device='gpu', name='Reader2', seed=0)
        # images on GPU
    else:
        images = fn.readers.numpy(file_root=data_file, random_shuffle=True,
            device='cpu', name='Reader', seed=0)
        labels = fn.readers.numpy(file_root=labels_file, random_shuffle=True,
            device='cpu', name='Reader2', seed=0)
        # images on CPU

    if device == 'gpu':
        images = fn.copy(images, device='gpu')
    
    # # entrypoint into hashing lib
    if sign_dataset and device == 'gpu':
        fn.python_function(images, batch_processing=True, function=lattice_hash, num_outputs=0)

    images = fn.random_resized_crop(images, size=[256, 256], device=device)
    images = fn.crop_mirror_normalize(
        images,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip(), device=device)
    return images, labels

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
model = model.cuda()

for device in ['gpu']:
    for sign_dataset in [False, True]:
        dali_loader = DALIGenericIterator(
            [get_dali_pipeline(
                batch_size=128,
                device=device,
                gpu_direct=False,
                sign_dataset=sign_dataset)],
            ['data', 'label'],
            reader_name='Reader'
        )

        t0 = time.monotonic()

        if sign_dataset and device=='cpu':
            sign(data_file, HashType.SHA256, Topology.SEQUENTIAL, InputType.FILES)

        for i, data in enumerate(dali_loader):
            x, y = data[0]['data'], data[0]['label']
            x = x.cuda()

        t1 = time.monotonic()

        for i, data in enumerate(dali_loader):
            x, y = data[0]['data'], data[0]['label']
            x = x.cuda()
            pred = model(x)
            loss = loss_func(pred, y)
            backward(loss, model)

        print(f'DALI loader on {device}, Sign: {sign_dataset}')
        print(f'Sign+Dataloader: {(t1-t0)*1000:.2f}, +Training: {(time.monotonic()-t0)*1000:.2f} ms')
