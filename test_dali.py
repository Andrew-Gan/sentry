from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import os
import time

# import src.sign as sign

# To run with different data, see documentation of nvidia.dali.fn.readers.file
# points to https://github.com/NVIDIA/DALI_extra
data_root_dir = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(data_root_dir, 'db', 'single', 'jpeg')


def loss_func(pred, y):
    pass


def model(x):
    pass


def backward(loss, model):
    pass


@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline():
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, name="Reader")
    # entrypoint into hashing lib
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

@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline_with_hashing():
    images, labels = fn.readers.file(
        file_root=images_dir, random_shuffle=True, name="Reader")
    # entrypoint into hashing lib
    print(images)
    # decode data on the GPU
    images = fn.decoders.image_random_crop(
        images, device="mixed", output_type=types.RGB)
    print(images)
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

dali_loader = DALIGenericIterator(
    [get_dali_pipeline(batch_size=16)],
    ['data', 'label'],
    reader_name='Reader'
)

dali_loader = DALIGenericIterator(
    [get_dali_pipeline_with_hashing(batch_size=16)],
    ['data', 'label'],
    reader_name='Reader'
)

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
model = model.to('cuda')

t0 = time.monotonic()
for i, data in enumerate(dali_loader):
    x, y = data[0]['data'], data[0]['label']
    pred = model(x)
    loss = loss_func(pred, y)
    backward(loss, model)
t1 = time.monotonic()

print(f'DALI loader: {(t1-t0)*1000:.2f} ms')

del model
