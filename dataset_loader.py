import os
import cv2
import torch
from torch.utils.data import Dataset

class VillageDataset(Dataset):

    def __init__(self, image_folder, mask_folder):

        self.samples = []

        for img in os.listdir(image_folder):

            name = os.path.splitext(img)[0]

            img_path = os.path.join(image_folder, img)
            mask_path = os.path.join(mask_folder, name + ".png")

            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        print("Valid samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        img_path, mask_path = self.samples[index]

        image = cv2.imread(img_path)
        image = cv2.resize(image,(512,512))
        image = image/255.0

        mask = cv2.imread(mask_path,0)
        mask = cv2.resize(mask,(512,512))
        mask = mask/255.0

        image = torch.tensor(image).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask
