from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import os
import time
import cupy as cp
import src.sign as sign
import FRL.dirichlet as dirichlet

SAMPLE_SIZE = 16

dataPath = os.path.join('dataset', 'cifar10', 'data')
labelPath = os.path.join('dataset', 'cifar10', 'labels')
sigPath = os.path.join('dataset', 'cifar10', 'signatures')

# TORCH HUB: load pretrained ML model and save to file
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
modelPath = './model.pth'
torch.save(model, modelPath)
model = model.cuda()

# SENTRY: precompile CUDA modules for Sentry operations
dataHasher = sign.build_hasher(sign.HashType.LATTICE, sign.Topology.ADD)
modelHasher = sign.build_hasher(sign.HashType.SHA256, sign.Topology.MERKLE)

def lattice_hash(data, sigs):
    partitionedBatch = {}
    for sample, sig in zip(data, sigs):
        sig = sig.item()
        if sig not in partitionedBatch:
            partitionedBatch[sig] = cp.expand_dims(sample, 0)
        else:
            partitionedBatch[sig] = cp.concatenate([partitionedBatch[sig], cp.expand_dims(sample, 0)])
    dataHasher.update(partitionedBatch, blockSize=data[0].nbytes)

# DALI
@pipeline_def(num_threads=8, device_id=0)
def get_dali_pipeline(device='gpu', gpu_direct=False):
    devRead = 'gpu' if gpu_direct else 'cpu'
    data = fn.readers.numpy(file_root=dataPath, random_shuffle=True,
        device=devRead, name='Reader1', seed=0)
    labels = fn.readers.numpy(file_root=labelPath, random_shuffle=True,
        device=devRead, name='Reader2', seed=0)
    sigs = fn.readers.numpy(file_root=sigPath, random_shuffle=True,
        device=devRead, name='Reader3', seed=0)

    if device == 'gpu':
        data = fn.copy(data, device='gpu')
        sigs = fn.copy(sigs, device='gpu')
        fn.python_function(data, sigs, batch_processing=True,
            function=lattice_hash, num_outputs=0)

    # data = fn.random_resized_crop(data, size=[256, 256], device=device)
    # data = fn.crop_mirror_normalize(
    #     data,
    #     crop_h=224,
    #     crop_w=224,
    #     mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    #     std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    #     mirror=fn.random.coin_flip(), device=device)

    return data, labels

def main():
    for num_partitions in [1<<i for i in range(10)]:
        print(f'Dataset partitions: {num_partitions}', flush=True)
        for dirichlet_alpha in [1]: # [0.1, 1, 10, 100, 1000, 10000, 100000]:
            print(f'Dirichlet alpha: {dirichlet_alpha}', flush=True)
            dirichlet.generate_signatures(labelPath, num_partitions, dirichlet_alpha, sigPath)

            for batch in [256]: #[32, 64, 128, 256, 512]:
                print(f'Batch: {batch}', flush=True)
                for device, gpu_direct in [('gpu', False)]: #[('cpu', False), ('gpu', False), ('gpu', True)]:
                    print(f'DALI: {device}, GDS: {gpu_direct}', flush=True)
                    dali_loader = DALIGenericIterator(
                        [get_dali_pipeline(
                            batch_size=batch,
                            device=device,
                            gpu_direct=gpu_direct)],
                        ['data', 'label'],
                        reader_name='Reader1'
                    )

                    t0 = time.monotonic()

                    for _ in range(SAMPLE_SIZE):
                        # if device == 'cpu':
                        #     sign.sign(os.path.join('dataset/cifar10'), sign.HashType.SHA256, sign.Topology.SEQUENTIAL, sign.InputType.FILES)
                        #     sign.sign(modelPath, HashType.SHA256, Topology.SEQUENTIAL, InputType.FILES)
                        # else:
                        #     sign.sign(model, HashType.SHA256, Topology.MERKLE, InputType.MODEL, modelHasher)

                        for i, data in enumerate(dali_loader):
                            x, y = data[0]['data'], data[0]['label']
                            # x = x.cuda()
                            # pred = model(x)

                    torch.cuda.synchronize()
                    t1 = time.monotonic()

                    print(f'Runtime: {(t1-t0)*1000/SAMPLE_SIZE:.2f} ms', flush=True)

main()
