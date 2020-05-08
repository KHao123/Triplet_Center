import os
import sys
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

'''
 train_0.txt --- mean : 0.49584001302719116 | std : 0.25280070304870605 | label0 : 4425 | label1 : 67 | label2 : 500
 train_1.txt --- mean : 0.49532830715179443 | std : 0.25283026695251465 | label0 : 4426 | label1 : 66 | label2 : 500
 test_0.txt --- mean : 0.49532830715179443 | std : 0.25283026695251465 | label0 : 4426 | label1 : 66 | label2 : 500
 test_1.txt --- mean : 0.49584001302719116 | std : 0.25280070304870605 | label0 : 4425 | label1 : 67 | label2 : 500
'''
_mean_ = [(0.49584,),(0.49532,)]
_std_ = [(0.252800,),(0.252830,)]
_normalize_ = {
    'train_0.txt' : transforms.Normalize(mean=(0.49584,),std=(0.252800,)) ,
    'train_1.txt' : transforms.Normalize(mean=(0.49532,),std=(0.252830,)) ,
    'test_0.txt' : transforms.Normalize(mean=(0.49532,),std=(0.252830,)) ,
    'test_1.txt' : transforms.Normalize(mean=(0.49584,),std=(0.252800,)) 
}

class CovidDataset(Dataset):
    def __init__(self, iterNo = 0, train = True, transform = None):
        self.train = train
        self.train_file = "train_{}.txt".format(iterNo)
        self.test_file = "test_{}.txt".format(iterNo)
        self.load_file = self.train_file if self.train else self.test_file
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                _normalize_[self.load_file]
            ]) if self.train else transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                _normalize_[self.load_file]
            ])
        
        self.targets = pd.read_csv(self.load_file).values
        if self.train:
            self.train_labels = torch.LongTensor([dta[1] for dta in self.targets])
        else:
            self.test_labels = torch.LongTensor([dta[1] for dta in self.targets])
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, label) where target is class_index of the target class.
        """
        path = self.targets[index][0]
        target = self.targets[index][1]
        # target = torch.LongTensor([target])
        # class_name = self.targets[index][2]
        img = Image.open(path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)




if __name__ == "__main__":
    dataset = CovidDataset(iterNo=1,train=False)
    print(dataset.__len__(),dataset.load_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for idx ,batch in enumerate(dataloader):
        (imgs, lbls) = batch
        print(imgs.size(), lbls)
        if idx>10:
            break 

