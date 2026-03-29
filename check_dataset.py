import os

images = os.listdir("dataset/train_images")
masks = os.listdir("dataset/masks")

print("Images:", len(images))
print("Masks:", len(masks))