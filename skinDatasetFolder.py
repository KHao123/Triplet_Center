import os
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from PIL import Image


def make_dataset(iterNo, data_dir):
    train_csv = 'split_data/origin_split_data/split_data_{}_fold_train.csv'.format(iterNo)
    fn = os.path.join(data_dir, train_csv)
    print('Loading train from {}'.format(fn))

    csvfile = pd.read_csv(fn)
    raw_train_data = csvfile.values
    train_data = []
    for x, y in raw_train_data:
        train_data.append((x, y))


    test_csv = 'split_data/origin_split_data/split_data_{}_fold_test.csv'.format(iterNo)
    fn = os.path.join(data_dir, test_csv)
    print('Loading test from {}'.format(fn))

    csvfile = pd.read_csv(fn)
    raw_test_data = csvfile.values
 
    test_data = []
    for x, y in raw_test_data:
        test_data.append((x, y))

    return train_data, test_data


class skinDatasetFolder(data.Dataset):
    '''
    原图大小（3， 450， 600）
    '''
    def __init__(self, train=True, iterNo=1, data_dir='../data'):
        
        self.train_data, self.test_data = make_dataset(iterNo,data_dir)
        self.train = train
        if self.train:
            self.train_labels = torch.LongTensor([dta[1] for dta in self.train_data])
        else:
            self.test_labels = torch.LongTensor([dta[1] for dta in self.test_data])

        mean , std = get_mean_std(iterNo)
        resize_img = 300
        img_size = 224
        normalized = transforms.Normalize(mean=mean,std=std)
        transform_train = transforms.Compose([
           transforms.Resize(resize_img),
           # transforms.RandomHorizontalFlip(),
           # transforms.RandomVerticalFlip(),
           # transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
           # transforms.RandomRotation([-180, 180]),
           # transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
           transforms.RandomCrop(img_size),
           transforms.ToTensor(),
           normalized
        #由于没有mean 和 std文件，这里先去掉normalized
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize_img),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalized
        ])
        self.transform = transform_train if self.train else transform_test

        raw_train_data = 'ISIC2018_Task3_Training_Input'
        self.train_data_dir = os.path.join(data_dir, raw_train_data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        if self.train:
            path, target = self.train_data[index]
        else:
            path, target = self.test_data[index]

        path = os.path.join(self.train_data_dir, path)
        imagedata = default_loader(path)

        if self.transform is not None:
            imagedata = self.transform(imagedata)

        return imagedata, target


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except ImportError:
        return pil_loader(path)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def get_mean_std(iterNo):
    filename = './mean_std.csv'
    csvfile = pd.read_csv(filename).values[int(iterNo)-1]
    print(csvfile)
    return csvfile[0:3], csvfile[3:]

if __name__ == '__main__':
    '''
    计算每类数据量
    num_dict = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    train_data, test_data = make_dataset(1,"/data/Public/Datasets/Skin7")
    target = [tup[1] for tup in train_data] 
    # print(test_data)
    # print(len(train_data) ,len(test_data))
    target.extend([tup[1] for tup in test_data])
    class_count = [target.count(i) for i in range(len(num_dict))]
    print(sum(class_count), len(target))
    if sum(class_count) == len(target):
        class_dict = {num_dict[i]:class_count[i] for i in range(len(num_dict))}
        print(class_dict) 
        
        #{'MEL':1113, 'NV':6705, 'BCC':514, 'AKIEC':327, 'BKL':1099, 'DF':115, 'VASC':142}
        
    '''
    get_mean_std(1)