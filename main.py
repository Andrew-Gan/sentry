from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import os
import time
import src.sign as sign
from src.sign import HashType, Topology, InputType
import dataset_formatter.formatter as formatter
import cupy as cp

# SENTRY: precompile CUDA modules for Sentry operations
torch.cuda.synchronize()
dataHasher = sign.compile_cuda_hasher(HashType.LATTICE, Topology.HOMOMORPHIC, InputType.DIGEST)
modelHasher = sign.compile_cuda_hasher(HashType.SHA256, Topology.MERKLE, InputType.MODULE)
# gpuSigner = sign.compile_cuda_signer()

def lattice_hash(data: list, metadata: list):
    partitions = {}
    for sample, (src, _) in zip(data, metadata):
        src = src.item()
        if src not in partitions:
            partitions[src] = []
        partitions[src].append(sample)
    dataHasher.update_dataset(partitions, blockSize=data[0].nbytes)

# TORCH HUB: load pretrained ML model and save to file
def get_model(model_name: list, device: str):
    if len(model_name) == 2:
        model = torch.hub.load(model_name[0], model_name[1], pretrained=True)
    elif len(model_name) == 3:
        model = torch.hub.load(model_name[0], model_name[1], model_name[2])

    modelPath = './model.pth'
    torch.save(model, modelPath)
    return model.to('cuda' if device=='gpu' else device), modelPath

def get_image_dataloader(dataPath: str, metaPath: str, batch: int, device: str, gds: bool):
    @pipeline_def(num_threads=8, device_id=0)
    def get_dali_pipeline_images(dataPath, metaPath, device='gpu', gds=False):
        data = fn.readers.numpy(file_root=dataPath, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader1', seed=0)
        metadata = fn.readers.numpy(file_root=metaPath, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader2', seed=0)
        
        if device == 'gpu' and not gds:
            data = fn.copy(data, device='gpu')
            metadata = fn.copy(metadata, device='gpu')

        if device == 'gpu':
            fn.python_function(data, metadata, batch_processing=True,
                function=lattice_hash, num_outputs=0, device='gpu')

        data = fn.random_resized_crop(data, size=[256, 256], device=device)
        data = fn.rotate(data, angle=fn.random.uniform(range=[0, 360]), device=device)
        data = fn.crop_mirror_normalize(
            data, crop_h=224, crop_w=224,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=fn.random.coin_flip(), device=device)

        return data, metadata

    dali_loader = DALIGenericIterator(
        [get_dali_pipeline_images(batch_size=batch, dataPath=dataPath,
            metaPath=metaPath, device=device, gds=gds)],
        ['data', 'label'],
        reader_name='Reader1'
    )

    return dali_loader, datasetPath

def get_text_dataloader(dataPath: str, metaPath: str, batch: int, device: str, gds: bool):
    @pipeline_def(num_threads=8, device_id=0)
    def get_dali_pipeline_texts(dataPath, metaPath, device='gpu', gds=False):
        data = fn.readers.numpy(file_root=dataPath, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader1', seed=0)
        metadata = fn.readers.numpy(file_root=metaPath, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader2', seed=0)

        if device == 'gpu' and not gds:
            data = fn.copy(data, device='gpu')
            metadata = fn.copy(metadata, device='gpu')

        if device == 'gpu':
            fn.python_function(data, metadata, batch_processing=True,
                function=lattice_hash, num_outputs=0, device='gpu')

        data = fn.pad(data, device=device)

        return data, metadata

    print(f'Batch: {batch}, DALI: {device}', flush=True)
    dali_loader = DALIGenericIterator(
            [get_dali_pipeline_texts(batch_size=batch, dataPath=dataPath,
                metaPath=metaPath, device=device, gds=gds)],
        ['data', 'label'],
        reader_name='Reader1'
    )

    return dali_loader, datasetPath

def inference(model: torch.nn.Module, dataloader: DALIGenericIterator):
    for data in dataloader:
        x, y = data[0]['data'], data[0]['label']
        # pred = model(x)
    torch.cuda.synchronize()
    return model

if __name__ == "__main__":
    device = 'gpu'
    gds = True
    model, modelPath = get_model(['pytorch/vision:v0.10.0', 'vgg19'], device=device)
    # model, modelPath = get_model(['huggingface/pytorch-transformers', 'model', 'bert-base-uncased'], device=device)

    datasetPath = os.path.join('dataset', 'cifar10')
    # datasetPath = os.path.join('dataset', 'hellaswag')

    # formatter.prepare_data(datasetPath, 'uoft-cs/cifar10', 4, 1)
    # formatter.prepare_data(datasetPath, 'Rowan/hellaswag', 4, 1)

    for batch in [32]:
        dataPath = os.path.join(datasetPath, 'data')
        metaPath = os.path.join(datasetPath, 'metadata')
        dataloader, datasetPath = get_image_dataloader(dataPath, metaPath, batch, device, gds)
        # dataloader, datasetPath = get_text_dataloader(dataPath, metaPath, batch, device, gds)

        model = inference(model, dataloader) # initialise overhead
        t0 = time.monotonic()

        model = inference(model, dataloader)

        t1 = time.monotonic()
        print(f'Model Training with DALI: {(t1-t0)*1000:.2f} ms\n', flush=True)

        del dataloader

    t2 = time.monotonic()

    if device == 'cpu':
        sign.sign(modelPath, HashType.SHA256, Topology.SERIAL, InputType.FILE)
    elif device == 'gpu':
        sign.sign(model, HashType.SHA256, Topology.MERKLE, InputType.MODULE,
            modelHasher)

    t3 = time.monotonic()
    print(f'Model Signing: {(t3-t2)*1000:.2f} ms\n', flush=True)

    if device == 'cpu':
        sign.sign(datasetPath, HashType.SHA256, Topology.SERIAL, InputType.FILE)
    elif device == 'gpu':
        sign.sign(dataHasher.compute(), HashType.LATTICE, Topology.HOMOMORPHIC, InputType.DIGEST)

    t4 = time.monotonic()
    print(f'Dataset Signing: {(t4-t3)*1000:.2f} ms\n', flush=True)

    del model

    print('Hashes of different dataset sources')
    print(dataHasher)
