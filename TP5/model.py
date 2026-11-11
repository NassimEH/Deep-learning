import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_cnn_features():
    resnet_pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    cnn_features = nn.Sequential(*list(resnet_pretrained.children())[:-2])
    for param in cnn_features.parameters():
        param.requires_grad = False
    return cnn_features, resnet_pretrained


class AttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.attention_layer = nn.Conv2d(feature_dim + hidden_dim, 1, 1)

    def forward(self, features, hidden_state):
        hidden_state_expanded = (
            hidden_state.unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, 1, features.size(2), features.size(3))
        )

        concat_features = torch.cat((features, hidden_state_expanded), dim=1)

        attention_scores = self.attention_layer(concat_features).squeeze(1)

        attention_weights = F.softmax(
            attention_scores.view(attention_scores.size(0), -1), dim=1
        )
        attention_weights = attention_weights.view(
            attention_scores.size(0),
            1,
            attention_scores.size(1),
            attention_scores.size(2),
        )

        context_vector = torch.sum(torch.sum(features * attention_weights, dim=3), dim=2)
        return context_vector, attention_weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_module, attention_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_module = attention_module

        total_dim = input_size + hidden_size + attention_size

        self.lin_i = nn.Linear(total_dim, hidden_size)
        self.lin_f = nn.Linear(total_dim, hidden_size)
        self.lin_c = nn.Linear(total_dim, hidden_size)
        self.lin_o = nn.Linear(total_dim, hidden_size)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, features, init_states=None):
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size, device=x.device)
            c_t = torch.zeros(bs, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            att_t, _ = self.attention_module(features, h_t)
            concat = torch.cat((att_t, x_t, h_t), dim=1)

            i_t = torch.sigmoid(self.lin_i(concat))
            f_t = torch.sigmoid(self.lin_f(concat))
            g_t = torch.tanh(self.lin_c(concat))
            o_t = torch.sigmoid(self.lin_o(concat))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)


class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, embedding_model, lstm_hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(*list(cnn_model.children())[:-2])
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.attention = AttentionModule(feature_dim=2048, hidden_dim=lstm_hidden_dim)
        self.embeddings = embedding_model
        vocab_size = embedding_model.weight.shape[0]
        emb_dim = embedding_model.weight.shape[1]

        self.lstm = LSTMWithAttention(emb_dim, lstm_hidden_dim, self.attention, 2048)
        self.fc_out = nn.Linear(lstm_hidden_dim, vocab_size)

    def forward(self, images, captions):
        cnn_features = self.cnn(images)
        emb = self.embeddings(captions)
        lstm_outputs, (hidden_state, _) = self.lstm(emb, cnn_features)
        output = self.fc_out(lstm_outputs)
        return output


if __name__ == "__main__":
    cnn_features, resnet = build_cnn_features()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = cnn_features(x)
    print(f"cnn_features output shape : {tuple(feats.shape)}")
