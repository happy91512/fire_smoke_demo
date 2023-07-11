import os, glob
import numpy as np
import cv2
import time
from typing import List

import tensorboard
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from efficientnet_pytorch import EfficientNet


from submodules.Useful_Tools_for_Me.FileTools.FileOperator import get_filenames

def read_file(img_dir: str, txt_dir: str, need_label = False) -> tuple[np.ndarray, list[int]]:
    """
    make label
    """
    img_files = sorted(get_filenames(img_dir, '*.jpg'))
    img_size = 128
    x = np.zeros((len(img_files), img_size, img_size, 3), dtype = np.uint8)
    y = np.zeros((len(img_files), 4), dtype=np.uint8)
    for i, file in enumerate(img_files):
        name = file.split('/')[-1]
        txt_path = os.path.join(txt_dir, name.replace('jpg', 'txt'))
        img = cv2.imread(file)
        x[i, :, :] = cv2.resize(img,(img_size, img_size))
        if need_label:
            f = open(txt_path, 'r')
            fire, smoke = 0, 0 
            for line in f.readlines():
                class_num = int(line[0])
                if class_num == 1:  #fire
                    fire = 1
                elif class_num == 3:  #smoke
                    smoke = 1
            y[i] = [int(not(fire or smoke)), int(fire == 1 and smoke == 0), int(fire == 0 and smoke == 1), int(fire and smoke)] #[nothing, fire, smoke, both]
    if need_label:
        return x, y
    else:
        return x

class MyDataset(Dataset):
    def __init__(self, x, y = None, transform = None):
        self.x = x
        self.y = y if y is None else  torch.LongTensor(y)
        self.transform = transform

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X   
         
    def __len__(self):
        return len(self.x)


     
# Set up dataset 



train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
    transforms.RandomRotation(15), # 隨機旋轉圖片
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])
config = {
    'num_epochs' : 3,
    'batch_size' : 128,
    'learning_rate' : 0.0003,
    'early_stop' : 400,
    'device' : "cuda:0" if torch.cuda.is_available() else "cpu"
}

dataset_dir = 'src/detect/dataset'
txt_dir = os.path.join(dataset_dir, 'yolotxt')
device = config['device']
print('device:', device)
if __name__ == '__main__':
    train_x, train_y = read_file(os.path.join(dataset_dir, 'train'), txt_dir, True)
    print(f'trainning data size : {len(train_y)}')
    val_x, val_y =  read_file(os.path.join(dataset_dir, 'val'), txt_dir, True)
    print(f'validation data size : {len(val_y)}')
    train_set = MyDataset(train_x, train_y, train_transform)
    val_set = MyDataset(val_x, val_y, test_transform)
    train_loader = DataLoader(dataset = train_set, batch_size = config['batch_size'], shuffle = True)
    val_loader = DataLoader(dataset = val_set, batch_size = config['batch_size'], shuffle = False)
    model = EfficientNet.from_name('efficientnet-b3')
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=4, bias=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])
    best_acc = 0
    #---------------------trainning---------------------
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        train_acc = []
        train_loss = []
        val_acc = []
        val_loss = []
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, (x, y) in loop:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            model = model.cuda()
            train_pred = model(x)
            train_pred = train_pred.to(device)
            y = torch.argmax(y, dim=1)
            batch_loss = loss(train_pred, y)
            batch_loss.backward()
            optimizer.step()
            acc = (train_pred.argmax(dim=-1) == y.to(device)).float().mean()
            train_acc.append(acc)
            train_loss.append(batch_loss.data.cpu())
            loop.set_description(f"Epoch[{epoch}/{config['num_epochs']}]")
            loop.set_postfix(train_loss = batch_loss.item(), train_acc = round(float(sum(train_acc)/len(train_acc)), 3))
            
    #---------------------validation---------------------
        model.eval()
        with torch.no_grad():
            loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
            for i, (x, y) in loop:
                x, y = x.to(config['device']), y.to(config['device'])
                val_pred = model(x)
                y = torch.argmax(y, dim=1)
                batch_loss = loss(val_pred, y)
                val_loss.append(batch_loss) 
                acc = (val_pred.argmax(dim=-1) == y.to(device)).float().mean()
                val_acc.append(acc)
                loop.set_description(f"Epoch[{epoch}/{config['num_epochs']}]")

                loop.set_postfix(val_loss = batch_loss.item(), val_acc = round(float(sum(val_acc)/len(val_acc)), 3))
            val_loss = sum(val_loss) / len(val_loss)
            val_acc = sum(val_acc) / len(val_acc)
            if val_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(model.state_dict(), 'src/detect/model/best.ckpt') # only save best to prevent output memory exceed error
                best_acc = val_acc
                early = 0
            else:
                early += 1
                if early > config['early_stop']:
                    print(f"No improvment {config['early_stop']} consecutive epochs, early stopping")
                    break 

#---------------------testing---------------------
# model_best = EfficientNet.from_name('efficientnet-b3')
# model_best._fc = nn.Linear(in_features=model._fc.in_features, out_features=4, bias=True)
# model_best.load_state_dict(torch.load('src/detect/model/best.ckpt'))

# test_x = read_file(os.path.join(dataset_dir, 'test'), txt_dir, False)
# print(f'testing data size : {len(test_x)}')
# test_set = MyDataset(x = test_x, transform = test_transform)
# test_loader = DataLoader(dataset = test_set, batch_size=32, shuffle=False)

# model.eval()
# pred = []
# loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
# with torch.no_grad():
#     for i, (data, _) in loop:
            