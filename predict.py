import torch
from torch.utils.data import DataLoader
from dataset_loader import VillageDataset
from model import VillageModel

dataset=VillageDataset("./dataset/train_images","./dataset/masks")

loader=DataLoader(dataset,batch_size=2,shuffle=True)

model=VillageModel()

loss_fn=torch.nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10):

    total_loss=0

    for images,masks in loader:

        preds=model(images)

        loss=loss_fn(preds,masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print("Epoch:",epoch,"Loss:",total_loss)

torch.save(model.state_dict(),"village_model.pth")