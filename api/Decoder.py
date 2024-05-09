import torch


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
