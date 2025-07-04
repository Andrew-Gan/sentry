import os
from common import get_model, get_image_dataloader
import sentry.signer

if __name__ == '__main__':
    model, _ = get_model('vgg19', pretrained=True, device='gpu')

    # dataloader, hasher = get_image_dataloader(
    #     data_path=os.path.join('dataset', 'cifar10', 'data'),
    #     meta_path=os.path.join('dataset', 'cifar10', 'metadata'),
    #     batch=128,
    #     device='gpu',
    #     gds=False,
    # )

    # for data in dataloader:
    #     x, y = data[0]['data'], data[0]['label']
        # pred = model(x)
    # print('[Trainer] Model training complete')

    sentry.signer.sign_model(model)
    print('[Trainer] Model signing complete')

    # sentry.signer.sign_dataset(hasher.compute())
    # print('[Trainer] Data set signing complete')
