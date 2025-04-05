
import os
import time
import src.sign as sign
from src.sign import HashType, Topology, InputType
from common import hash_batch, get_model, get_image_dataloader, get_text_dataloader

if __name__ == '__main__':
    batch = 128
    device = 'gpu'
    gds = False
    num_sources = 16
    dataset_path = os.path.join('dataset', 'cifar10')
    # dataset_path = os.path.join('dataset', 'hellaswag')

    if device == 'cpu':
        signer_model, serializer_model = sign.build(
            HashType.SHA256, Topology.SERIAL, InputType.FILE)
    elif device == 'gpu':
        _, signer_model, serializer_model = sign.build(
            HashType.SHA256, Topology.MERKLE, InputType.MODULE)
        hash_batch.hasher, signer_dataset, serializer_dataset = sign.build(
            HashType.LATTICE, Topology.HOMOMORPHIC, InputType.DIGEST, num_sources)

    model, modelPath = get_model(['pytorch/vision:v0.10.0', 'vgg19'], pretrained=False, device=device)
    # model, modelPath = get_model(['huggingface/pytorch-transformers', 'model', 'bert-base-uncased'], device=device)

    data_path = os.path.join(dataset_path, 'data')
    meta_path = os.path.join(dataset_path, 'metadata')

    dataloader = get_image_dataloader(data_path, meta_path, batch, device, gds)
    # dataloader = get_text_dataloader(data_path, meta_path, batch, device, gds)

    # start = time.monotonic()

    # for data in dataloader:
    #     x, y = data[0]['data'], data[0]['label']
    #     # pred = model(x)

    # print(f'[Trainer] Model Training: {(time.monotonic()-start)*1000:.2f} ms\n', flush=True)
    start = time.monotonic()

    if device == 'cpu':
        sign.sign_item(modelPath, signer_model, serializer_model, InputType.FILE)
    elif device == 'gpu':
        sign.sign_item(model, signer_model, serializer_model, InputType.MODULE)

    print(f'[Trainer] Model Signing: {(time.monotonic()-start)*1000:.2f} ms\n', flush=True)
    start = time.monotonic()

    # if device == 'cpu':
    #     sign.sign_item(dataset_path, signer_dataset, serializer_dataset, InputType.FILE)
    # elif device == 'gpu':
    #     sign.sign_item(hash_batch.hasher.compute(), signer_dataset, serializer_dataset, InputType.DIGEST)

    # print(f'[Trainer]: Dataset Signing: {(time.monotonic()-start)*1000:.2f} ms\n', flush=True)

    # if device == 'gpu':
    #     print('Hashes of different dataset sources')
    #     print(hash_batch.hasher)
