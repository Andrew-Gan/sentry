import os
import sentry
from common import get_model, get_image_dataloader
import sentry.verifier

if __name__ == '__main__':
    model, _ = get_model(['pytorch/vision:v0.10.0', 'vgg19'], pretrained=True)

    dataloader, hasher = get_image_dataloader(
        data_path=os.path.join('dataset', 'cifar10', 'data'),
        meta_path=os.path.join('dataset', 'cifar10', 'metadata'),
        batch=128,
        device='gpu',
        gds=True,
    )

    for data in dataloader:
        x, y = data[0]['data'], data[0]['label']
        # pred = model(x)

    dataset_digest = hasher.compute()
    print('[Inferencer] Model inference complete')

    sentry.verifier.verify_model(model)
    print('[Inferencer] Model verification complete')

    sentry.verifier.verify_dataset(dataset_digest)
    print('[Inferencer] Data set verification complete')
