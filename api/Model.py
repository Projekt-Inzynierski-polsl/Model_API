import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import CNNEncoder


class CRNN(nn.Module):
    def __init__(self, size, num_chars, num_channels, device, dims, num_layers, cnn_model=1):
        super(CRNN, self).__init__()
        self.size = size
        self.num_chars = num_chars + 1
        self.device = device
        self.num_channels = num_channels
        self.dims = dims
        self.num_layers = num_layers
        self.cnn_model = cnn_model

        self.cnn_encoder = CNNEncoder.CNNEncoder(self.size, self.num_channels, self.dims, self.cnn_model, self.device)

        # decoder part
        self.rnn = nn.GRU(self.dims, self.dims // 2, bidirectional=True, num_layers=self.num_layers, batch_first=True)
        self.output = nn.Linear(self.dims, self.num_chars)

    def __ctc_loss(self, features, targets, target_lengths):
        input_lengths = torch.full(size=(features.size(1),), fill_value=features.size(0), dtype=torch.int32)
        loss = nn.CTCLoss(blank=0)(features, targets, input_lengths, target_lengths)
        return loss

    def forward(self, images, targets, target_lengths):
        cnn_output = self.cnn_encoder(images)
        rnn_output, _ = self.rnn(cnn_output)
        linear_output = self.output(rnn_output)
        x = linear_output.permute(1, 0, 2)
        x = torch.nn.functional.log_softmax(x, 2)
        # loss = self.__ctc_loss(x, targets, target_lengths)
        # return x, loss
        return x
