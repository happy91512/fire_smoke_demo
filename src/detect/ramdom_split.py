import random
import os
import shutil
import torch
from torch.utils.data import random_split

def split(src_dir: str, train_val_ratio: float, test_size = 0): 
    files = os.listdir(src_dir)
    train_val_size = len(files) - test_size
    val_size = int(train_val_ratio * train_val_size)
    train_size = train_val_size - val_size
    print(f'total size: {len(files)} train size: {train_size} val size: {val_size} test size: {test_size}')
    train_set , val_set, test_set = random_split(
    dataset=files,
    lengths=[train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(0)
    )
    l = ['train/', 'val/', 'test/']
    for i, set in enumerate([train_set , val_set, test_set]):
        for file in set:
            # print(file, l[i] + file)
            shutil.copyfile(src_dir + '/' + file, 'src/detect/dataset' + '/'+ l[i] + file)

split('src/detect/dataset/images', 0.2, 200)