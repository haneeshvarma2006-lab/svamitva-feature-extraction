import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset_loader import VillageDataset
from model import VillageModel

dataset = VillageDataset(
    "dataset/train_images",
    "dataset/masks"
)

loader = DataLoader(dataset,batch_size=4,shuffle=True)

model = VillageModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 30

for epoch in range(epochs):

    total_loss = 0

    for img,mask in loader:

        pred = model(img)

        loss = criterion(pred,mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Loss:",total_loss)

torch.save(model.state_dict(),"village_model.pth")