from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import os
import time
import src.sign as sign
import dataset_formatter.formatter as formatter
import cupy as cp

# SENTRY: precompile CUDA modules for Sentry operations
torch.cuda.synchronize()
dataHasher = sign.build_hasher(sign.HashType.LATTICE, sign.Topology.HOMOMORPHIC, sign.InputType.DATASET)
modelHasher = sign.build_hasher(sign.HashType.SHA256, sign.Topology.MERKLE, sign.InputType.MODEL)

def lattice_hash(data, metadata):
    partitions = {}
    for sample, (src, _) in zip(data, metadata):
        sample = cp.asarray(sample)
        if src not in partitions:
            partitions[src] = []
        partitions[src].append(sample)
    dataHasher.update_dataset(partitions, blockSize=data[0].nbytes)

def get_image_model_dataset():
    # TORCH HUB: load pretrained ML model and save to file
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    modelPath = './model.pth'
    torch.save(model, modelPath)
    model = model.cuda()

    datasetPath = os.path.join('dataset', 'cifar10')
    dataPath=os.path.join(datasetPath, 'data')
    metaPath=os.path.join(datasetPath, 'metadata')
    formatter.prepare_data(datasetPath, 'uoft-cs/cifar10', 4, 1)

    @pipeline_def(num_threads=8, device_id=0)
    def get_dali_pipeline_images(dataPath, metaPath, device='gpu'):
        data = fn.readers.numpy(file_root=dataPath, random_shuffle=True,
                                device='cpu', name='Reader1', seed=0)
        metadata = fn.readers.numpy(file_root=metaPath, random_shuffle=True,
                                device='cpu', name='Reader2', seed=0)

        if device == 'gpu':
            fn.python_function(data, metadata, batch_processing=True,
                function=lattice_hash, num_outputs=0, device='cpu')
            data = fn.copy(data, device='gpu')

        data = fn.random_resized_crop(data, size=[256, 256], device=device)
        data = fn.rotate(data, angle=fn.random.uniform(range=[0, 360]), device=device)
        data = fn.crop_mirror_normalize(
            data, crop_h=224, crop_w=224,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=fn.random.coin_flip(), device=device)

        return data, metadata

    batch=32
    device='gpu'
    print(f'Batch: {batch}, DALI: {device}', flush=True)
    dali_loader = DALIGenericIterator(
        [get_dali_pipeline_images(batch_size=batch, dataPath=dataPath,
            metaPath=metaPath, device=device,)],
        ['data', 'label'],
        reader_name='Reader1'
    )

    t0 = time.monotonic()

    if device == 'cpu':
        sign.sign(datasetPath, sign.HashType.SHA256, sign.Topology.SEQUENTIAL, sign.InputType.FILE)
        sign.sign(modelPath, sign.HashType.SHA256, sign.Topology.SEQUENTIAL, sign.InputType.FILE)
    else:
        sign.sign(model, sign.HashType.SHA256, sign.Topology.MERKLE, sign.InputType.MODEL, modelHasher)

    for data in dali_loader:
        x, y = data[0]['data'], data[0]['label']
        # x = x.cuda()
        # pred = model(x)

    torch.cuda.synchronize()
    t1 = time.monotonic()

    print(f'DALI runtime: {(t1-t0)*1000:.2f} ms\n', flush=True)

def get_llm_model_dataset():
    # TORCH HUB: load pretrained ML model and save to file
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    modelPath = './model.pth'
    torch.save(model, modelPath)
    model = model.cuda()

    datasetPath = os.path.join('dataset', 'hellaswag')
    dataPath=os.path.join(datasetPath, 'data')
    metaPath=os.path.join(datasetPath, 'metadata')
    formatter.prepare_data(datasetPath, 'Rowan/hellaswag', 4, 1)

    @pipeline_def(num_threads=8, device_id=0)
    def get_dali_pipeline_texts(dataPath, metaPath, device='gpu'):
        data = fn.readers.numpy(file_root=dataPath, random_shuffle=True,
                                device='cpu', name='Reader1', seed=0)
        metadata = fn.readers.numpy(file_root=metaPath, random_shuffle=True,
                                device='cpu', name='Reader2', seed=0)

        if device == 'gpu':
            fn.python_function(data, metadata, batch_processing=True,
                function=lattice_hash, num_outputs=0, device='cpu')
            data = fn.copy(data, device='gpu')

        data = fn.pad(data, device=device)

        return data, metadata

    batch = 128
    device='gpu'
    print(f'Batch: {batch}, DALI: {device}', flush=True)
    dali_loader = DALIGenericIterator(
            [get_dali_pipeline_texts(batch_size=batch, dataPath=dataPath,
                metaPath=metaPath, device=device,)],
        ['data', 'label'],
        reader_name='Reader1'
    )

    t0 = time.monotonic()

    if device == 'cpu':
        sign.sign(datasetPath, sign.HashType.SHA256, sign.Topology.SEQUENTIAL, sign.InputType.FILE)
        sign.sign(modelPath, sign.HashType.SHA256, sign.Topology.SEQUENTIAL, sign.InputType.FILE)
    else:
        sign.sign(model, sign.HashType.SHA256, sign.Topology.MERKLE, sign.InputType.MODEL, modelHasher)

    for data in dali_loader:
        x, y = data[0]['data'], data[0]['label']
        # x = x.cuda()
        # pred = model(x)

    torch.cuda.synchronize()
    t1 = time.monotonic()

    print(f'DALI runtime: {(t1-t0)*1000:.2f} ms\n', flush=True)

# get_image_model_dataset()
get_llm_model_dataset()
print('Hashes of different dataset sources')
print(dataHasher)
sign.sign(dataHasher.compute(), sign.HashType.LATTICE, sign.Topology.HOMOMORPHIC, sign.InputType.DATASET)
