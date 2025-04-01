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

def lattice_hash(data: list, metadata: list):
    partitions = {}
    for sample, (src, _) in zip(data, metadata):
        src = src.item()
        if src not in partitions:
            partitions[src] = []
        partitions[src].append(sample)
    lattice_hash.data_hasher.update_dataset(partitions, blockSize=data[0].nbytes)

# TORCH HUB: load pretrained ML model and save to file
def get_model(model_name: list, device: str):
    if len(model_name) == 2:
        model = torch.hub.load(model_name[0], model_name[1], pretrained=True)
    elif len(model_name) == 3:
        model = torch.hub.load(model_name[0], model_name[1], model_name[2])

    modelPath = './model.pth'
    torch.save(model, modelPath)
    return model.to('cuda' if device=='gpu' else device), modelPath

def get_image_dataloader(data_path: str, meta_path: str, batch: int, device: str, gds: bool):
    @pipeline_def(num_threads=8, device_id=0)
    def get_dali_pipeline_images():
        data = fn.readers.numpy(file_root=data_path, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader1', seed=0)
        metadata = fn.readers.numpy(file_root=meta_path, random_shuffle=True,
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
        [get_dali_pipeline_images(batch_size=batch)],
        ['data', 'label'],
        reader_name='Reader1'
    )

    return dali_loader

def get_text_dataloader(data_path: str, meta_path: str, batch: int, device: str, gds: bool):
    @pipeline_def(num_threads=8, device_id=0)
    def get_dali_pipeline_texts():
        data = fn.readers.numpy(file_root=data_path, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader1', seed=0)
        metadata = fn.readers.numpy(file_root=meta_path, random_shuffle=True,
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
        [get_dali_pipeline_texts(batch_size=batch)],
        ['data', 'label'],
        reader_name='Reader1'
    )

    return dali_loader

def inference(model: torch.nn.Module, dataloader: DALIGenericIterator):
    for data in dataloader:
        x, y = data[0]['data'], data[0]['label']
        # pred = model(x)
    torch.cuda.synchronize()
    return model

if __name__ == "__main__":
    batch = 128
    device = 'gpu'
    gds = False
    num_sources = 128

    data_hasher = None

    if device == 'cpu':
        signer_cpu, serializer_cpu = sign.build(HashType.SHA256, Topology.SERIAL, InputType.FILE)
    elif device == 'gpu':
        # SENTRY: precompile CUDA modules for Sentry operations
        torch.cuda.synchronize()
        data_hasher = sign.compile_cuda_hasher(HashType.LATTICE, Topology.HOMOMORPHIC,InputType.DIGEST)
        model_hasher = sign.compile_cuda_hasher(HashType.SHA256, Topology.MERKLE, InputType.MODULE)

        # signer_merkle_gpu, serializer_merkle_gpu = sign.build(HashType.SHA256, Topology.MERKLE, InputType.MODULE, model_hasher)
        signer_lattice_gpu, serializer_lattice_gpu = sign.build(HashType.LATTICE,
            Topology.HOMOMORPHIC, InputType.DIGEST, num_sigs=num_sources)

    model, modelPath = get_model(['pytorch/vision:v0.10.0', 'vgg19'], device=device)
    # model, modelPath = get_model(['huggingface/pytorch-transformers', 'model', 'bert-base-uncased'], device=device)

    dataset_path = os.path.join('dataset', 'cifar10')
    # dataset_path = os.path.join('dat.aset', 'hellaswag')

    formatter.prepare_data(dataset_path, 'uoft-cs/cifar10', num_sources, 1)
    # formatter.prepare_data(dataset_path, 'Rowan/hellaswag', num_sources, 1)

    data_path = os.path.join(dataset_path, 'data')
    meta_path = os.path.join(dataset_path, 'metadata')

    lattice_hash.data_hasher = data_hasher
    dataloader = get_image_dataloader(data_path, meta_path, batch, device, gds)
    # dataloader = get_text_dataloader(data_path, meta_path, batch, device, gds, data_hasher)

    # model = inference(model, dataloader) # initialise overhead
    t0 = time.monotonic()

    model = inference(model, dataloader)

    t1 = time.monotonic()
    print(f'Model Training with DALI: {(t1-t0)*1000:.2f} ms\n', flush=True)

    if device == 'cpu':
        sign.sign(modelPath, signer_cpu, serializer_cpu, InputType.FILE)
    elif device == 'gpu':
        sign.sign(model, signer_merkle_gpu, serializer_merkle_gpu, InputType.MODULE)

    t2 = time.monotonic()
    print(f'Model Signing: {(t2-t1)*1000:.2f} ms\n', flush=True)

    if device == 'cpu':
        sign.sign(dataset_path, signer_cpu, serializer_cpu, InputType.FILE)
    elif device == 'gpu':
        sign.sign(data_hasher.compute(), signer_lattice_gpu, serializer_lattice_gpu, InputType.DIGEST)

    t3 = time.monotonic()
    print(f'Dataset Signing: {(t3-t2)*1000:.2f} ms\n', flush=True)

    del dataloader
    del model

    print('Hashes of different dataset sources')
    print(data_hasher)
