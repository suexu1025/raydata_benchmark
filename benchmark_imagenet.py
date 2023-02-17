import ray
path = "gs://pytorch-datasets/imagenet"

import gcsfs
import glob
import os
import tensorflow.io as io
from ray.data.block import BlockMetadata
from ray.data.datasource import FileMetadataProvider,FastFileMetadataProvider
import time
import numpy as np
import pprint

def _rand_crop(image, label):
    low_x=low_y=low_z=0
    high_x=high_y=high_z=128
    image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
    label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
    return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

def load_data(path, files_pattern):
    data = sorted(io.gfile.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data

from torch.utils.data import Dataset,DataLoader
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.experimental.pjrt as pt
from torch.utils.data.distributed import DistributedSampler
import torchvision
from PIL import Image

class PytTrain(Dataset):
    def __init__(self, images, dataset, **kwargs):
        self.dataset = dataset
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with io.gfile.GFile(os.path.join(self.dataset, self.images[idx]), 'rb') as f:
            data = {"image": Image.open(f)}
        #data = self.rand_crop(data)
        #data = self.train_transforms(data)     
        return data["image"]

def ray_loader(paths_x):
    device = xm.xla_device()
    provider=FastFileMetadataProvider()
    ds = ray.data.read_numpy(paths_x,filesystem=gcsfs.GCSFileSystem(), meta_provider=provider, parallelism = 4)
    ds.to_torch()

    start = time.time()
    for j in range(10):
        for i, batch in enumerate(ds.iter_batches(batch_size=1)):
            batch = torch.as_tensor(batch[0])
            batch = xm.send_cpu_data_to_device(batch, device)
            batch.to(device)
            pass

    training_time = (time.time() - start)/10
    print(f"Training time for ray : {training_time:.2f} seconds")

def ray_loader_(local_rank, ds):
    device = xm.xla_device()
    ds.to_torch()

    start = time.time()
    for j in range(10):
        for i, batch in enumerate(ds.iter_batches(batch_size=1)):
            batch = torch.as_tensor(batch[0])
            batch = xm.send_cpu_data_to_device(batch, device)
            batch.to(device)
            pass

    training_time = (time.time() - start)/10
    print(f"Training time for ray : {training_time:.2f} seconds")

def torch_dataloader(paths, world_size):
        device = xm.xla_device()
        with io.gfile.GFile(os.path.join(paths, 'imagenetindex_train.json') as f:
            path_x = json.load(f)
        paths = os.path.join(paths, 'train')
        paths_x = [name.split('/')[-2:] for name in paths_x]
        local_rank = xm.get_ordinal()

        train_dataset = PytTrain(paths_x, paths)
        from pprint import pprint
        pprint(local_rank)
        pprint(world_size)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True
        )

        train_loader = pl.MpDeviceLoader(train_loader, device)

        start = time.time()
        num = 0
        for j in range(10):
            for i, batch in enumerate(train_loader):
                batch[0].to(device)
                num = num + 1
                pass

        training_time = (time.time() - start)/10
        print(f"Training time for {num} images pytorch: {training_time:.2f} seconds")

@ray.remote
class LoaderWorker:
    def __init__(self, rank: int):
        pt._initialize_multiprocess(rank, 4)
        pass

    def load(self, paths_x, world_size: int) -> int:
        print("worldsize")
        pprint.pprint(world_size)
        torch_dataloader(paths_x, world_size)
        return 0


def ray_main(flags):
    path = os.path.join(flags.data_dir, "train")
    paths_x = load_data(path, "*.JPEG")

    # num of worker per host
    num_process = 4
    workers = [LoaderWorker.remote(i) for i in range(num_process)]
    features_ref = ray.put(paths_x)

    world_size = flags.world #xm.xrt_world_size()
    ray.get([w.load.remote(features_ref, world_size) for w in workers])


import torch_xla.distributed.xla_multiprocessing as xmp

def xla_main(local_rank, flags):
    paths = flags.data_dir

    world_size = xm.xrt_world_size()
    print("worldsize")
    pprint.pprint(world_size)
    if flags.loader == "torch":
        torch_dataloader(paths, world_size)
    else:
        ray_loader(paths)

    xm.rendezvous("exit")

@ray.remote
def consume(data) -> int:
    start = time.time()
    for j in range(10):
        num_batches = 0
        for batch in data.iter_batches(batch_size=1):
            num_batches += 1
    training_time = (time.time() - start)/10
    print(f"Training time for ray : {training_time:.2f} seconds")
    return num_batches

@ray.remote
class Worker:
    def __init__(self, rank: int):
        pt._initialize_multiprocess(rank, 4)
        pass

    def train(self, shard) -> int:
        local_rank = xm.get_ordinal()
        from pprint import pprint
        pprint(local_rank)

        device = xm.xla_device()
        num_batches = 0
        start = time.time()
        for j in range(10):
            for batch in shard.iter_batches(batch_size=1):
                batch = torch.as_tensor(batch[0])
                batch = xm.send_cpu_data_to_device(batch, device)
                batch.to(device)            
                num_batches += 1
                pass
        training_time = (time.time() - start)/10
        print(f"Training time {num_batches} of images for ray :  {training_time:.2f} seconds")
        return shard.count()


import argparse
import os

PARSER = argparse.ArgumentParser(description="benchmark dataloader")
PARSER.add_argument('-mp', '--mp', dest='mp',  choices=["xla", "ray"], default="xla")
PARSER.add_argument('-loader', '--loader', dest='loader',  choices=["torch", "ray"], default="torch")
PARSER.add_argument('-world_size', '--world_size', dest='world',  type=int, default=4)
#PARSER.add_argument('-data_dir', '--data_dir', dest='data_dir',  type=str, default="gs://mlperf-dataset/data/2021_Brats_np/11_3d")
PARSER.add_argument('-data_dir', '--data_dir', dest='data_dir',  type=str, default="gs://pytorch-datasets/imagenet")
import numpy
if __name__ == '__main__':
    flags = PARSER.parse_args()
    if flags.mp == 'ray' and flags.loader == 'ray':
        path = os.path.join(flags.data_dir, "train")
        paths_x = load_data(path, "*.JPEG")
        host = flags.world // 4
        num_per_host = len(paths_x) // host
        print(num_per_host)
        paths_x = numpy.random.choice(paths_x, size = num_per_host).tolist()
        print(len(paths_x))
        provider=FastFileMetadataProvider()
        ds = ray.data.read_images(paths_x,filesystem=gcsfs.GCSFileSystem(), meta_provider=provider)
        workers = [Worker.remote(i) for i in range(4)]

        shards = ds.split(n=4, locality_hints=workers)
        ray.get([w.train.remote(s) for w, s in zip(workers, shards)])

        #print(ray.get(consume.remote(ds)))
    elif flags.mp == 'ray':
        ray.init(ignore_reinit_error=True)
        ray_main(flags)
    elif flags.mp == 'xla':
        print("using mode 3 \n")
        xmp.spawn(xla_main,  args=(flags,))
    elif flags.mp == 'xla' and flags.loader == 'ray':
        print("using mode 4 \n")
        path = os.path.join(flags.data_dir, "train")
        paths_x = load_data(path, "*_x.npy")

        provider=FastFileMetadataProvider()
        ds = ray.data.read_numpy(paths_x,filesystem=gcsfs.GCSFileSystem(), meta_provider=provider)
        xmp.spawn(ray_loader_,  args=(ds, ))
    elif flags.mp == 'ray' and flags.loader == 'ray':
        print("using mode 5 \n")
        path = os.path.join(flags.data_dir, "train")
        paths_x = load_data(path, "*_x.npy")

        provider=FastFileMetadataProvider()
        ds = ray.data.read_numpy(paths_x,filesystem=gcsfs.GCSFileSystem(), meta_provider=provider)

        workers = [Worker.remote(i) for i in range(4)]

        shards = ds.split(n=4, locality_hints=workers)

        ray.get([w.train.remote(s) for w, s in zip(workers, shards)])
    else:
        pass