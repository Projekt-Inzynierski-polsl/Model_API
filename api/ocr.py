import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from torchvision.models import resnet18, ResNet18_Weights


class CNNEncoder(nn.Module):
    def __init__(self, size, num_channels, dims, cnn_model, device):
        super(CNNEncoder, self).__init__()
        self.size = size
        self.num_channels = num_channels
        self.dims = dims
        self.cnn_model = cnn_model
        self.device = device

        if self.cnn_model == 0:
            self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
            if self.num_channels == 1:
                self.cnn.conv1 = nn.Conv2d(
                    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
            self.encode = self.rnn_encoder

        cnn_input_size = self.encode(
            torch.rand(1, self.num_channels, self.size[0], self.size[1])
        )

        self.linear = nn.Sequential(
            nn.Linear(cnn_input_size.shape[-1], self.dims), nn.GELU(), nn.Dropout(0.5)
        )

    def rnn_encoder(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.linear(x)
        return x


class CRNN(nn.Module):
    def __init__(
        self, size, num_chars, num_channels, device, dims, num_layers, cnn_model=1
    ):
        super(CRNN, self).__init__()
        self.size = size
        self.num_chars = num_chars + 1
        self.device = device
        self.num_channels = num_channels
        self.dims = dims
        self.num_layers = num_layers
        self.cnn_model = cnn_model

        self.cnn_encoder = CNNEncoder(
            self.size, self.num_channels, self.dims, self.cnn_model, self.device
        ).to(device=self.device)

        # decoder part
        self.rnn = nn.GRU(
            self.dims,
            self.dims // 2,
            bidirectional=True,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(self.dims, self.num_chars)

    def __ctc_loss(self, features, targets, target_lengths):
        input_lengths = torch.full(
            size=(features.size(1),), fill_value=features.size(0), dtype=torch.int32
        )

        loss = nn.CTCLoss(blank=0)(features, targets, input_lengths, target_lengths)
        return loss

    def forward(self, images):
        cnn_output = self.cnn_encoder(images)
        rnn_output, _ = self.rnn(cnn_output)
        linear_output = self.output(rnn_output)
        x = linear_output.permute(1, 0, 2)
        x = torch.nn.functional.log_softmax(x, 2)
        # loss = self.__ctc_loss(x, targets, target_lengths)
        return x


class Decoder:
    def __init__(self, decode_type):
        self.decode_type = decode_type  # greedy or beam search

    def decode(self, list_predictions, target_classes):
        if self.decode_type == 0:
            return self.greedy_decoder(list_predictions, target_classes)
        else:
            pass

    def greedy_decoder(self, list_predictions, target_classes):
        dict_class = dict(zip(range(len(target_classes)), target_classes))
        decoded_predictions = []
        for i in list_predictions:
            batch_predictions = i.permute(1, 0, 2)
            batch_predictions = torch.softmax(batch_predictions, 2)
            batch_predictions = torch.argmax(batch_predictions, 2)
            batch_predictions = batch_predictions.detach().cpu().numpy()

            for j in range(batch_predictions.shape[0]):
                temp_text = " "
                is_space = False
                for k in range(batch_predictions.shape[1]):
                    if batch_predictions[j][k] == 0:
                        is_space = True
                    else:
                        tmp_char = dict_class[batch_predictions[j][k] - 1]
                        if tmp_char == temp_text[-1] and is_space is True:
                            temp_text += tmp_char
                        elif tmp_char != temp_text[-1]:
                            temp_text += tmp_char
                        is_space = False
                decoded_predictions.append(temp_text.strip())
        return decoded_predictions


def make_transcription(image):

    # Create model and load weights
    height = 100
    width = 800
    num_channels = 1
    num_layers = 3
    dims = 256
    size = (height, width)
    device = "cpu"
    num_chars = 79

    model = CRNN(
        size=size,
        num_chars=num_chars,
        num_channels=num_channels,
        device=device,
        dims=dims,
        num_layers=num_layers,
        cnn_model=0,
    ).to(device)

    path = "./OCR_model.pt"
    model.load_state_dict(torch.load(path))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tmp_region = Image.fromarray(image)
    tmp_region = tmp_region.resize((800, 100), resample=Image.BILINEAR)
    tmp_region = transform(tmp_region)
    tmp_region = tmp_region.unsqueeze(0).to(device)

    output = model(tmp_region)

    decoder = Decoder(0)
    classes = [
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ":",
        ";",
        "?",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "|",
    ]

    decoded_output = decoder.decode([output], classes)

    return decoded_output
