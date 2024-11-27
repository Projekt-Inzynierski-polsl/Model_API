import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image, ImageOps


def make_segmentation(image):

    # Create model and load weights
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        classes=1,
        activation="sigmoid",
        in_channels=1,
    )

    model_path = "./segmentation_model.pt"
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model = model.to("cpu")
    model.eval()

    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    size = [224, 224]

    image = Image.fromarray(image)
    tmp_img = image.convert("L")
    input_shape = np.array(tmp_img).shape

    threshold_value = 150
    tmp_img = tmp_img.point(lambda p: 255 if p > threshold_value else 0)
    tmp_img = ImageOps.invert(tmp_img)
    tmp_img = tmp_img.resize((size[1], size[0]))

    tmp_img = np.array(tmp_img)
    tmp_img = transform(tmp_img)

    # Run model for image

    output = model(tmp_img.unsqueeze(0).to("cpu"))
    output = output[0].detach().numpy().squeeze()
    output = cv2.resize(
        output, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    output = np.array(255 * np.array(output), np.uint8)

    return output
