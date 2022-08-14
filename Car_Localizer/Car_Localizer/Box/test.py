from box import draw_box
import torch
import cv2
import torchvision.transforms as transforms
import numpy
from PIL import Image

image = cv2.imread("car.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define a transform to convert the image to tensor
transform_tensor = transforms.ToTensor()

# Convert the image to PyTorch tensor
tensor = transform_tensor(image)

transform_pil = transforms.ToPILImage()

pil_img = transform_pil(tensor)

data = numpy.array(pil_img)

draw_box(data, 256, 1148, 1400, 752)

img = Image.fromarray(data, "RGB")
img.show()
