from torch.utils.data import Dataset
from torchvision.transforms import *
from PIL import Image
import torch
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class LoadData(Dataset):
    def __init__(self):
        super(LoadData, self).__init__()
        self.path_imgs = self.get_img("data")
    
    def __getitem__(self, index):
        return torch.randn(100), self.transform(self.path_imgs[index])

    def __len__(self):
        return len(self.path_imgs)
    
    def get_img(self, path):
        list_img = []
        for split in os.listdir(path):
            inner_path = os.path.join(path, split)
            for class_name in os.listdir(inner_path):
                class_path = os.path.join(inner_path, class_name)
                for img in os.listdir(class_path):
                    img_path = os.path.join(class_path, img)
                    list_img.append(img_path)
        return list_img
    
    def transform(self, img):
        img = Image.open(img)
        transform = Compose([
            ToTensor(),
            Normalize(mean=0.5, std=0.5),
        ])
        img = transform(img)
        return torch.flatten(img)
    

