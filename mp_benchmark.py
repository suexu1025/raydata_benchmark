import ray
path = "gs://mlperf-dataset/data/2021_Brats_np/11_3d"

import gcsfs
import glob
import os
import tensorflow.io as io
from ray.data.block import BlockMetadata
from ray.data.datasource import FileMetadataProvider,FastFileMetadataProvider
import time
import numpy as np
  

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

class PytTrain(Dataset):
    def __init__(self, images, labels, dataset, **kwargs):
        self.dataset = dataset
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with io.gfile.GFile(os.path.join(self.dataset, self.images[idx]), 'rb') as f, io.gfile.GFile(os.path.join(self.dataset, self.labels[idx]), 'rb') as g:
            data = {"image": np.load(f), "label": np.load(g)}
        #data = self.rand_crop(data)
        #data = self.train_transforms(data)     
        return data["image"], data["label"]

paths_x = load_data(path, "*_x.npy")
paths_y = load_data(path, "*_y.npy")

def ray_loader(paths_x):
    device = xm.xla_device()
    provider=FastFileMetadataProvider()
    ds = ray.data.read_numpy(paths_x,filesystem=gcsfs.GCSFileSystem(), meta_provider=provider)
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

def torch_dataloader(paths_x, paths_y):
        device = xm.xla_device()
        paths_x = [name.split('/')[-1] for name in paths_x]
        paths_y = [name.split('/')[-1] for name in paths_y]
        train_dataset = PytTrain(paths_x, paths_y, path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True
        )

        train_loader = pl.MpDeviceLoader(train_loader, device)

        start = time.time()
        for j in range(10):
            for i, batch in enumerate(train_loader):
                batch[0].to(device)
                pass

        training_time = (time.time() - start)/10
        print(f"Training time for pytorch: {training_time:.2f} seconds")

def ray_main(flags):
    path = "gs://mlperf-dataset/data/2021_Brats_np/11_3d"
    paths_x = load_data(path, "*_x.npy")
    paths_y = load_data(path, "*_y.npy")

    @ray.remote
    def data_loading(paths_x, paths_y, idx, flags):
        if flags.loader == "torch":
            torch_dataloader(paths_x, paths_y)
        else:
            ray_loader(paths_x)
    
    features_ref = ray.put(paths_x)
    label_ref = ray.put(paths_y)
    task_ref = []
    for i in range(4):
        task_ref.append(data_loading.remote(features_ref,label_ref, i, flags))
        
    result = ray.get(task_ref)

import torch_xla.distributed.xla_multiprocessing as xmp

def xla_main(local_rank, flags):
    path = "gs://mlperf-dataset/data/2021_Brats_np/11_3d"
    paths_x = load_data(path, "*_x.npy")
    paths_y = load_data(path, "*_y.npy")
    if flags.loader == "torch":
        torch_dataloader(paths_x, paths_y)
    else:
        ray_loader(paths_x)

    xm.rendezvous("exit")

import argparse
import os

PARSER = argparse.ArgumentParser(description="benchmark dataloader")
PARSER.add_argument('--mp', dest='multi-process runner',  choices=["xla", "ray"], default="xla")
PARSER.add_argument('--loader', dest='loader type',  choices=["torch", "ray"], default="torch")

if __name__ == '__main__':
    flags = PARSER.parse_args()
    flags.mp = 'xla'
    flags.loader = 'torch'
    if flags.mp == 'ray':
        ray.init(ignore_reinit_error=True)
        ray_main(flags)
    elif flags.mp == 'xla':
        xmp.spawn(xla_main,  args=(flags,))
    else:
        pass