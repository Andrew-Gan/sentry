import pathlib
from common import get_model, get_image_dataloader
import sentry
from huggingface_hub import login
from sentry.model_signing.hashing.topology import *

import time

if __name__ == '__main__':
    with open('hf_access_token', 'r') as f:
        login(token=f.read().rstrip())

    for modelName in ['resnet152', 'bert', 'vgg19', 'gpt2', 'gpt2-xl']:
        model = get_model(modelName, pretrained=True, device='gpu')
        for hashAlgo in HashAlgo:
            for topology in Topology:
                if topology == Topology.SERIAL:
                    continue
                for workflow in Workflow:
                    if topology == Topology.LATTICE and hashAlgo != HashAlgo.BLAKE2XB:
                        continue
                    if workflow == Workflow.LAYERED_SORTED and topology != Topology.LATTICE:
                        continue
                    filename = f'{modelName}-{topology.name}-{workflow.name}-{hashAlgo.name}.sig'
                    print(filename, flush=True)
                    sentry.verify_model(model, filename)
    print('[Inferencer] Model verification complete')

    dataloader, hasher = get_image_dataloader(
        path=pathlib.Path('dataset/cifar10'),
        batch=128,
        device='gpu',
        gds=False,
    )

    start = time.perf_counter()
    for data in dataloader:
        x, y = data[0]['data'], data[0]['label']
        # pred = model(x)
    end = time.perf_counter()
    print(f'[Inferencer] Dataset runtime: {1000*(end-start):.2f} ms')

    sentry.verify_dataset(hasher.compute())
    print('[Inferencer] Dataset verification complete')
