import os
from common import get_model, get_image_dataloader, get_text_dataloader
from sentry.model_signing.hashing.topology import HashAlgo, Topology
import sentry.signer
from huggingface_hub import login
import time

if __name__ == '__main__':
    with open('hf_access_token', 'r') as f:
        login(token=f.read())
        model = get_model('vgg19', pretrained=True, device='gpu')

        dataloader, hasher = get_text_dataloader(
            data_path=os.path.join('dataset', 'cifar10', 'data'),
            meta_path=os.path.join('dataset', 'cifar10', 'metadata'),
            batch=128,
            device='gpu',
            gds=False,
        )

        start = time.perf_counter()
        for data in dataloader:
            x, y = data[0]['data'], data[0]['label']
            # pred = model(x)
        end = time.perf_counter()
        print(f'{1000*(end-start):.2f}')
        print('[Trainer] Model training complete')

        # sentry.signer.sign_model(model, HashAlgo.SHA256, Topology.MERKLE_INPLACE)
        # print('[Trainer] Model signing complete')

        # sentry.signer.sign_dataset(hasher.compute())
        # print('[Trainer] Data set signing complete')
