import os
from skinDatasetFolder import make_dataset
import csv
from PIL import Image
import torchvision.transforms as transforms

def cal_skin7():
     data_dir = '/data/Public/Datasets/Skin7'
     raw_train_data = 'ISIC2018_Task3_Training_Input'
     train_data_dir = os.path.join(data_dir, raw_train_data)
     transform1 = transforms.Compose([
	    transforms.ToTensor(), 
	    ]
    )
     with open("./mean_std.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['mean1', 'mean2', 'mean3', 'std1', 'std2', 'std3'])
        for iterNo in range(5):
            mean_list = [0,0,0]
            std_list = [0,0,0]
            train_data, _ = make_dataset(iterNo+1, data_dir)
            
            for path, _ in train_data:
                path = os.path.join(train_data_dir, path)
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    img.convert('RGB')
                    
                    img = transform1(img).numpy().squeeze()
                    # print(img[0,:,:].mean())
                    # print(img.shape)
                mean_list = [mean_list[i] + img[i,:,:].mean() for i in range(3)]
                # print(mean_list)
                std_list = [std_list[i] + img[i,:,:].std() for i in range(3)]
                
            mean_list = [mean / len(train_data) for mean in mean_list]
            std_list = [std / len(train_data) for std in std_list]
            # mean_list.append(image_mean / len(train_data))
            # std_list.append(image_std / len(train_data)
            print(mean_list,std_list)
            mean_list.extend(std_list)
            csvwriter.writerow(mean_list)




if __name__ =="__main__":
    cal_skin7()

