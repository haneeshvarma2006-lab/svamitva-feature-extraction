import os
import cv2
import torch
from torch.utils.data import Dataset

class VillageDataset(Dataset):

    def __init__(self,image_dir,mask_dir):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.images=os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):

        img_path=os.path.join(self.image_dir,self.images[idx])
        mask_path=os.path.join(self.mask_dir,self.images[idx].replace(".tif",".png"))

        image=cv2.imread(img_path)
        image=cv2.resize(image,(512,512))

        mask=cv2.imread(mask_path,0)
        mask=cv2.resize(mask,(512,512))

        image=image/255.0

        image=torch.tensor(image).permute(2,0,1).float()
        mask=torch.tensor(mask).unsqueeze(0).float()

        return image,mask