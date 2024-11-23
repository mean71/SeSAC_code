import matplotlib.pyplot as plt
import numpy as np
# import torch 
# import torch.nn as nn 
# import torch.optim as optim
# import matplotlib.pyplot as plt

# from torch.utils import DataLoader
# from time import time 
# from typing import List 

# from conv2d_mine import CNN 
# from data_handler import train_data, train_loader, small_train_loader


# Get one batch of images
train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True) 

data_iter = iter(train_loader)
images, labels = next(data_iter)

for image in images:
    print(image.shape)
    print(image)
    break

# # Class names in CIFAR-10
# class_names = train_data.classes

# # Function to display images in a grid
# def imshow(img):
#     img = img / 2 + 0.5  # Unnormalize (if normalized)
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert from tensor image

# # Display the images
# plt.figure(figsize=(10, 10))  # Set the figure size
# grid_img = torchvision.utils.make_grid(images, nrow=4)
# imshow(grid_img)
# plt.title("Sample Images from CIFAR-10")
# plt.axis('off')
# plt.show()
