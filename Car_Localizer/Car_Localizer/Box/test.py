from box import draw_box
import torch
import cv2
import torchvision.transforms as transforms

image = cv2.imread("car.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor
tensor = transform(image)

draw_box(tensor, 200, 200, 200, 200)
