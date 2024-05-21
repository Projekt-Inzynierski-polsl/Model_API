import cv2
import torch
from torchvision import transforms
from PIL import Image
from sklearn import preprocessing
import numpy as np
import re

import Model
import Decoder


def change_output(decoded_line):
    decoded_line = decoded_line[0].replace("|", " ").strip()
    decoded_line = re.sub(r'\s+([,.:;?!])', r'\1', decoded_line)

    return decoded_line


def encode_labels(targets):
    labels = [[y for y in x] for x in targets]
    labels_flat = [i for label_list in labels for i in label_list]
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels_flat)
    labels_encoded = [label_encoder.transform(x) for x in labels]

    labels_lengths = [len(x) for x in labels_encoded]
    padding_symbol = -1
    max_length = max(len(seq) for seq in labels_encoded)
    labels_encoded = [list(label) + [padding_symbol] * (max_length - len(label)) for label in labels_encoded]
    labels_encoded = np.array(labels_encoded) + 1

    return labels_encoded, labels_lengths, label_encoder.classes_


def call_model(segmented_image):
    device = "cpu"
    num_layers = 3
    dims = 256
    cnn_model = 0

    height = 100
    width = 800
    num_channels = 1

    size = (height, width)

    crnn = Model.CRNN(size=size, num_chars=79, num_channels=num_channels, device=device, dims=dims,
                      num_layers=num_layers, cnn_model=cnn_model).to(device)

    model_path = "./300.pt"
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    crnn.load_state_dict(state_dict)
    crnn_cpu = crnn.to('cpu')

    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    image = Image.fromarray(segmented_image).convert("RGB")
    image = Image.eval(image, lambda x: 255 - x)
    image = image.resize((size[1], size[0]), resample=Image.BILINEAR)

    image = transform(image)
    image_with_batch = image.unsqueeze(0).to('cpu')
    targets, targets_lengths, target_classes = encode_labels(["target"])
    targets = torch.tensor(targets, dtype=torch.long).to('cpu')
    # print(image_with_batch.shape)
    # print(targets)
    # print(targets_lengths)
    crnn_cpu.eval()
    output = crnn_cpu(image_with_batch, targets, targets_lengths)
    decoder = Decoder.Decoder(0)
    classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
               'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z', '|']
    dec_output = decoder.decode([output], classes)
    dec_output1 = [change_output(dec_output)]
    return dec_output1
