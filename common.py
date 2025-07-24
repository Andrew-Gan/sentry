from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
from sentry.model_signing.hashing.topology import *
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torchvision

def hash_batch(data: list, metadata: list):
    partitions = {}
    for sample, (src, _) in zip(data, metadata):
        src = src.item()
        if src not in partitions:
            partitions[src] = []
        partitions[src].append(sample)
    hash_batch.hasher.update_dataset(partitions, blockSize=data[0].nbytes)

# TORCH HUB: load pretrained ML model and save to file
def get_model(model_name: list, pretrained: bool = False, device: str = 'gpu'):
    model_name = model_name.lower()
    if model_name == 'resnet152':
        model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-152")
    elif model_name == 'bert':
        model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    elif model_name == 'vgg19':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'gpt2':
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    elif model_name == 'gpt2-xl':
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")
    else:
        raise NotImplementedError(f'Fetching of {model_name} not implemented')

    return model.to('cuda' if device=='gpu' else device)

def get_image_dataloader(data_path: str, meta_path: str, batch: int, device: str, gds: bool):
    if device == 'cpu':
        raise NotImplementedError('Lattice hashing on CPU not supported yet')
    elif device == 'gpu':
        hash_batch.hasher = HomomorphicGPU(HashAlgo.LATTICE, InputType.DIGEST)

    @pipeline_def(num_threads=8, device_id=0)
    def get_dali_pipeline_images():
        data = fn.readers.numpy(file_root=data_path, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader1', seed=0)
        metadata = fn.readers.numpy(file_root=meta_path, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader2', seed=0)
        
        if device == 'gpu':
            if not gds:
                data = fn.copy(data, device='gpu')
                metadata = fn.copy(metadata, device='gpu')
            fn.python_function(data, metadata, batch_processing=True,
                function=hash_batch, num_outputs=0, device='gpu')

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

    return dali_loader, hash_batch.hasher

def get_text_dataloader(data_path: str, meta_path: str, batch: int, device: str, gds: bool):
    if device == 'cpu':
        raise NotImplementedError('Lattice hashing on CPU not supported yet')
    elif device == 'gpu':
        hash_batch.hasher = HomomorphicGPU(HashAlgo.LATTICE, InputType.DIGEST)

    @pipeline_def(num_threads=8, device_id=0)
    def get_dali_pipeline_texts():
        data = fn.readers.numpy(file_root=data_path, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader1', seed=0)
        metadata = fn.readers.numpy(file_root=meta_path, random_shuffle=True,
            device='cpu' if not gds else device, name='Reader2', seed=0)

        if device == 'gpu' and not gds:
            data = fn.copy(data, device='gpu')
            metadata = fn.copy(metadata, device='gpu')

        data = fn.pad(data, device=device)

        if device == 'gpu':
            fn.python_function(data, metadata, batch_processing=True,
                function=hash_batch, num_outputs=0, device='gpu')

        return data, metadata

    dali_loader = DALIGenericIterator(
        [get_dali_pipeline_texts(batch_size=batch)],
        ['data', 'label'],
        reader_name='Reader1'
    )

    return dali_loader, hash_batch.hasher
