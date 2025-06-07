
import os
from common import get_model, get_image_dataloader, HashType, Topology
from transformers import AutoImageProcessor, AutoModelForImageClassification
import sentry.signer

# ('pytorch/vision:v0.10.0', 'resnet152'),
# ('huggingface/pytorch-transformers', 'model', 'bert-base-uncased'),
# ('huggingface/transformers', 'modelForCausalLM', 'gpt2'),
# ('pytorch/vision:v0.10.0', 'vgg19'),
# ('huggingface/transformers', 'modelForCausalLM', 'gpt2-large'),
# ('huggingface/transformers', 'modelForCausalLM', 'gpt2-xl'),

if __name__ == '__main__':
    model, _ = get_model(('pytorch/vision:v0.10.0', 'resnet152'))

    dataloader, hasher = get_image_dataloader(
        data_path=os.path.join('dataset', 'cifar10', 'data'),
        meta_path=os.path.join('dataset', 'cifar10', 'metadata'),
        batch=128,
        device='gpu',
        gds=False,
    )

    for data in dataloader:
        x, y = data[0]['data'], data[0]['label']
    #     pred = model(x)

    print('[Trainer] Model training complete')

    sentry.signer.sign_model(model, HashType.SHA256, Topology.MERKLE)
    print('[Trainer] Model signing complete')

    sentry.signer.sign_dataset(hasher.compute())
    print('[Trainer] Data set signing complete')
