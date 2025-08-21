import os
from common import get_model, get_image_dataloader
import sentry
from huggingface_hub import login

if __name__ == '__main__':
    with open('hf_access_token', 'r') as f:
        login(token=f.read().rstrip())

    model = get_model('vgg19', pretrained=True, device='gpu')

    # dataloader, hasher = get_image_dataloader(
    #     data_path=os.path.join('dataset', 'cifar10', 'data'),
    #     meta_path=os.path.join('dataset', 'cifar10', 'metadata'),
    #     batch=128,
    #     device='gpu',
    #     gds=True,
    # )

    # for data in dataloader:
    #     x, y = data[0]['data'], data[0]['label']
        # pred = model(x)
    # print('[Trainer] Model training complete')

    sentry.sign_model(model)
    print('[Trainer] Model signing complete')

    # sentry.sign_dataset(hasher.compute())
    # print('[Trainer] Data set signing complete')
