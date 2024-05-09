import os

import cv2
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import numpy as np


def segment_image(img):
    # define model with loss function
    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", classes=1,
                     activation="sigmoid")
    model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                    padding=(3, 3), bias=False)

    criterion = torch.nn.BCEWithLogitsLoss()

    # load pretrained weights and set model to evaluation
    model_path = "./model_30.pt"
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to("cpu")
    model.eval()

    # image options
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    size = [224, 224]

    image = Image.fromarray(img).convert("RGB")

    input_size = [np.array(image).shape[0], np.array(image).shape[1]]

    image = image.resize((size[1], size[0]), resample=Image.BILINEAR)
    image = transform(image)
    image = image.unsqueeze(0).to('cpu')

    output = model(image)[0].detach().numpy().squeeze()

    output = Image.fromarray(output)
    output = output.resize((input_size[1], input_size[0]), resample=Image.BILINEAR)
    output = np.array(255 * np.array(output), np.uint8)  # jeśli jako cv2
    return output
