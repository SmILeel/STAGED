from datetime import datetime
import os
from pathlib import Path


_device = None


def get_save_path(save_path, comments):
    dir_path = save_path
    now = datetime.now()
    datestr = now.strftime('%m_%d-%H:%M:%S')

    paths = [
        f'./pretrained/{dir_path}/{comments}/best_{datestr}.pt',
        f'./results/{dir_path}/{comments}/{datestr}.csv',
    ]

    for path in paths:
        dirname = os.path.dirname(path)
        Path(dirname).mkdir(parents=True, exist_ok=True)

    return paths


def set_device(dev):
    global _device
    _device = dev


def get_device():
    return _device

