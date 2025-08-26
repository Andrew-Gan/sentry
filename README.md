# Sentry

An end-to-end GPU framework for authenticating machine learning artifacts.



![](sentry.png "Sentry")

<!-- markdown-toc --bullets="-" -i README.md -->

<!-- toc -->

- [Overview](#overview)
- [Projects](#projects)
  - [Model Signing](#model-signing)
  - [SLSA for ML](#slsa-for-ml)
- [Status](#status)
- [Contributing](#contributing)

<!-- tocstop -->

## Overview

This work is described in an accepted paper to be published soon. Stay tuned for more.  
Before setting up, please refer to [link](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) to ensure compatibility between driver, compiler and runtime APIs.

## Setup

### Docker
[Docker](https://docs.docker.com/get-started/get-docker/)  
[Docker Compose](https://docs.docker.com/compose/install/)   
[Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```
mkdir -p ./signatures
```

### Native run
```
python3 -m venv .venv
source .venv/bin/activate

export TORCH_HOME=./torch

pip install -r requirements.txt

openssl ecparam -name prime256v1 -genkey -noout -out private.pem
openssl ec -in private.pem -pubout -out public.pem

export CUFILE_ENV_PATH_JSON=cufile.json

nvcc -Xcompiler '-fPIC' -o ./RapidEC/gsv.so -shared ./RapidEC/gsv.cu

mkdir -p ./signatures
```

### Huggingface login
Some ML models from huggingface require a user to be logged in to avoid timeout errors.  
Create an account and follow the [guide](https://huggingface.co/docs/hub/en/security-tokens) to acquire your access token.  
Then, place the access token in the file named 'hf_access_token'.

## Run

### Docker
```
docker compose up --build sentry_dataset
docker compose up --build sentry_trainer
docker compose up --build sentry_inferencer
```

### Native run
```
python agent_dataset.py uoft-cs/cifar10 16 1 dataset/cifar10
python agent_trainer.py --sig_out ./signatures --model_path ./torch private-key --private_key private.pem
python agent_inferencer.py --sig_out ./signatures --model_path ./torch private-key --private_key private.pem
```

## Example
Before Sentry:
```python
import torchvision.models as models
from torch.utils.data import DataLoader

model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
dataloader = DataLoader(testing_data, batch_size=128, shuffle=True)

for data in dataloader:
    x, y = data[0]['data'], data[0]['label']
    pred = model(x)
```

After Sentry:
```python
import torchvision.models as models
from common import get_image_dataloader
import sentry

model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
# get Sentry's custom DALI-based dataloader which supports GPUDirect and dataset hashing
dataloader, hasher = get_image_dataloader(
    path='./dataset/cifar10', batch=128, device='gpu', gds=True,
)

# verify model
sentry.verify_model(model)

for data in dataloader:
    x, y = data[0]['data'], data[0]['label']
    pred = model(x)

# verify dataset
sentry.verify_dataset(hasher.compute())
```

## Evaluation

This project demonstrates how to protect the integrity of a model by signing it
with [Sigstore](https://www.sigstore.dev/), a tool for making code signatures
transparent without requiring management of cryptographic key material.

When users download a given version of a signed model they can check that the
signature comes from a known or trusted identity and thus that the model hasn't
been tampered with after training.

We are able to sign large models with very good performance, as the following
table shows:

| Model                         | Size  |  Hash Time |
|-------------------------------|-------|:----------:|
| microsoft/resnet-152          | 270M  | 5.5 ms     |
| google-bert/bert-base-uncased | 538M  | 11.4 ms    |
| pytorch/vision/vgg19          | 1.1G  | 29.1 ms    |
| openai-community/gpt2         | 1.1G  | 32.3 ms    |
| openai-community/gpt2-xl      | 8.6G  | 296.6 ms   |


## Contributions

### Adding a new CUDA kernel for hashing
`cd sentry/model_signing/cuda`
`cd sentry/model_signing/hashing/topology.py`
