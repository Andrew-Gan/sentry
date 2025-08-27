import pathlib
from common import get_model, get_image_dataloader
import sentry
from huggingface_hub import login
from sentry.model_signing.hashing.topology import *

if __name__ == '__main__':
    with open('hf_access_token', 'r') as f:
        login(token=f.read().rstrip())

    model = get_model('vgg19', pretrained=True, device='gpu')
    
    sentry.verify_model(model)
    print('[Inferencer] Model verification complete')

    dataloader, hasher = get_image_dataloader(
        path=pathlib.Path('dataset/cifar10'),
        batch=128,
        device='gpu',
        gds=False,
    )

    for data in dataloader:
        x, y = data[0]['data'], data[0]['label']
        # pred = model(x)

    sentry.verify_dataset(hasher.compute())
    print('[Inferencer] Dataset verification complete')
