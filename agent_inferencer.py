import pathlib
from common import get_model, get_image_dataloader
import sentry
from huggingface_hub import login

if __name__ == '__main__':
    with open('hf_access_token', 'r') as f:
        login(token=f.read().rstrip())

    model = get_model('vgg19', pretrained=True, device='gpu')

    dataloader, hasher = get_image_dataloader(
        path=pathlib.Path('dataset/cifar10'),
        batch=128,
        device='gpu',
        gds=False,
    )

    for data in dataloader:
        x, y = data[0]['data'], data[0]['label']
        # pred = model(x)
    print('[Inferencer] Model inference complete')

    sentry.verify_dataset(hasher.compute())
    print('[Inferencer] Data set verification complete')

    sentry.verify_model(model)
    print('[Inferencer] Model verification complete')
