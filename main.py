from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import os
import time

from src.sign import sign_gpu, HashType, Topology, InputType

# To run with different data, see documentation of nvidia.dali.fn.readers.file
# points to https://github.com/NVIDIA/DALI_extra
data_root_dir = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(data_root_dir, 'db', 'single', 'jpeg')
npy_dir = os.path.join(data_root_dir, 'db', '3D', 'MRI', 'Knee', 'npy_3d', 'STU00001')


def loss_func(pred, y):
    pass


def model(x):
    pass


def backward(loss, model):
    pass


@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline_cpu():
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, name="Reader")
    # entrypoint into hashing lib
    # digest = sign_gpu(images, HashType.SHA256, Topology.MERKLE, InputType.GPU)
    # print(f'{digest.hex()}')
    # decode data on the GPU
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB)
    # the rest of processing happens on the GPU as well
    images = fn.resize(images, resize_x=256, resize_y=256)
    images = fn.crop_mirror_normalize(
        images,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip())
    return images, labels

dali_loader_cpu = DALIGenericIterator(
    [get_dali_pipeline_cpu(batch_size=16)],
    ['data', 'label'],
    reader_name='Reader'
)

@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline_gpu():
    images = fn.readers.numpy(
        file_root=npy_dir, random_shuffle=True, device='gpu')
    # decode data on the GPU
    images = fn.random_resized_crop(
        images, size=[256, 256], device='gpu')
    # the rest of processing happens on the GPU as well
    images = fn.crop_mirror_normalize(
        images,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip(), device='gpu')
    return images, 0

dali_loader_gpu = DALIGenericIterator(
    [get_dali_pipeline_gpu(batch_size=16)],
    ['data', 'label'],
)

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
model = model.to('cuda')

# hash model

t0 = time.monotonic()
for i, data in enumerate(dali_loader):
    x, y = data[0]['data'], data[0]['label']
    pred = model(x)
    loss = loss_func(pred, y)
    backward(loss, model)
t1 = time.monotonic()

print(f'DALI loader: {(t1-t0)*1000:.2f} ms')

del model
