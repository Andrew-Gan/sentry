from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import os
import time
import src.sign as sign
import FRL.cifar10 as cifar10
import cupy as cp

datasetPath = os.path.join('dataset', 'cifar10')
dataPath = os.path.join(datasetPath, 'data')
metaPath = os.path.join(datasetPath, 'metadata')

# TORCH HUB: load pretrained ML model and save to file
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
modelPath = './model.pth'
torch.save(model, modelPath)
model = model.cuda()

# SENTRY: precompile CUDA modules for Sentry operations
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

@pipeline_def(num_threads=8, device_id=0)
def get_dali_pipeline(device='gpu'):
    data = fn.readers.numpy(file_root=dataPath,
        random_shuffle=True, device='cpu', name='Reader1', seed=0)
    metadata = fn.readers.numpy(file_root=metaPath,
        random_shuffle=True, device='cpu', name='Reader2', seed=0)

    if device == 'gpu':
        fn.python_function(data, metadata, batch_processing=True,
            function=lattice_hash, num_outputs=0, device='cpu')
        data = fn.copy(data, device='gpu')

    data = fn.random_resized_crop(data, size=[256, 256], device=device)
    data = fn.rotate(data, angle=fn.random.uniform(range=[0, 360]), device=device)
    data = fn.crop_mirror_normalize(
        data,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip(), device=device)

    return data, metadata

def main():
    for numPartitions in [1]:
        for dirichletAlpha in [1]:
            print(f'Dataset partitions: {numPartitions}, Dirichlet alpha: {dirichletAlpha}', flush=True)
            cifar10.prepare_cifar10(datasetPath, numPartitions, dirichletAlpha)

            for batch in [128]:
                print(f'Batch: {batch}', flush=True)
                for device in ['gpu']:
                    print(f'DALI: {device}', flush=True)
                    dali_loader = DALIGenericIterator(
                        [get_dali_pipeline(
                            batch_size=batch,
                            device=device,)],
                        ['data', 'label'],
                        reader_name='Reader1'
                    )

                    t0 = time.monotonic()

                    if device == 'cpu':
                        sign.sign(datasetPath, sign.HashType.SHA256, sign.Topology.SEQUENTIAL, sign.InputType.FILE)
                        sign.sign(modelPath, sign.HashType.SHA256, sign.Topology.SEQUENTIAL, sign.InputType.FILE)
                    else:
                        sign.sign(model, sign.HashType.SHA256, sign.Topology.MERKLE, sign.InputType.MODEL, modelHasher)

                    for i, data in enumerate(dali_loader):
                        x, y = data[0]['data'], data[0]['label']
                        x = x.cuda()
                        pred = model(x)

                    torch.cuda.synchronize()
                    t1 = time.monotonic()

                    print(f'DALI runtime: {(t1-t0)*1000:.2f} ms\n', flush=True)

main()
print('Hashes of different dataset sources')
print(dataHasher)
