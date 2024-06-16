# -*- coding: gbk -*-
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


def is_uchar(uchar):
    return (
        u'\u4e00' <= uchar <= u'\u9fa5' or
        u'\u0030' <= uchar <= u'\u0039' or
        u'\u0041' <= uchar <= u'\u005a' or
        u'\u0061' <= uchar <= u'\u007a' or
        uchar in {'，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'}
    )



with open(r'.\corpus_utf8\雪山飞狐.txt', encoding='utf8', errors='ignore') as f:
    data = f.readlines()

pattern = re.compile(r'\(.*\)')
data = [pattern.sub('', lines) for lines in data]
data = [line.replace('……', '。') for line in data if len(line) > 1]
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)


vocab = list(set(data))
char2id = {c: i for i, c in enumerate(vocab)}
id2char = {i: c for i, c in enumerate(vocab)}
numdata = [char2id[char] for char in data]



class TextDataset(Dataset):
    def __init__(self, data, time_steps):
        self.data = torch.tensor(data, dtype=torch.long)
        self.time_steps = time_steps

    def __len__(self):
        return len(self.data) // self.time_steps

    def __getitem__(self, idx):
        start = idx * self.time_steps
        x = self.data[start:start + self.time_steps]
        y = self.data[start + 1:start + self.time_steps + 1]
        return x, y


def create_data_loader(data, batch_size, time_steps):
    dataset = TextDataset(data, time_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.mha2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        attn1, _ = self.mha1(x, x, x, attn_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)
        attn2, _ = self.mha2(out1, enc_output, enc_output, attn_mask=padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.layernorm3(out2 + ffn_output)


# 完整的Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, num_layers, num_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                 dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, num_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, num_model)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(num_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(num_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.final_layer = nn.Linear(num_model, target_vocab_size)

    def forward(self, inp, tar, training=False):
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder_embedding(inp)
        enc_output = enc_output.permute(1, 0, 2)  # Shape to (S, N, E) for PyTorch MultiheadAttention

        for i in range(len(self.encoder_layers)):
            enc_output = self.encoder_layers[i](enc_output, None)

        dec_output = self.decoder_embedding(tar)
        dec_output = dec_output.permute(1, 0, 2)

        for i in range(len(self.decoder_layers)):
            dec_output = self.decoder_layers[i](dec_output, enc_output, look_ahead_mask, None)

        dec_output = dec_output.permute(1, 0, 2)
        final_output = self.final_layer(dec_output)
        return final_output

    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp).to(inp.device)
        dec_padding_mask = self.create_padding_mask(inp).to(inp.device)
        look_ahead_mask = self.create_look_ahead_mask(tar.size(1)).to(tar.device)
        dec_target_padding_mask = self.create_padding_mask(tar).to(tar.device)
        combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    def create_padding_mask(self, seq):
        return (seq == 0).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones((size, size)), diagonal=1)
        return mask == 1


num_layers = 8
num_model = 128
dff = 256
num_heads = 8

input_vocab_size = len(vocab)+2
target_vocab_size = len(vocab)+2
dropout_rate = 0.1

batch_size = 64
time_steps = 50
epochs = 100
learning_rate = 0.001

model = TransformerModel(num_layers, num_model, num_heads, dff, input_vocab_size, target_vocab_size, time_steps,
                         time_steps, dropout_rate)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_data = create_data_loader(numdata, batch_size, time_steps)


class LossHistory:
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, loss):
        self.losses.append(loss)


history = LossHistory()

losses = []
model.train()
for epoch in range(epochs):
    total_loss = 0
    for i, (inp, tar) in enumerate(train_data):
        inp, tar = inp.to('cuda' if torch.cuda.is_available() else 'cpu'), tar.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        output = model(inp, tar[:, :-1])
        loss = criterion(output.view(-1, target_vocab_size), tar[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_data)
    history.on_epoch_end(average_loss)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')


plt.plot(history.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transformer Model Training Loss')
plt.show()


def generate_text(model, start_string, num_generate=100):
    input_eval = [char2id[s] for s in start_string]
    input_eval = torch.tensor(input_eval, dtype=torch.long).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    text_generated = []
    decoder_input = torch.tensor([[char2id['。']]], dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        for _ in range(num_generate):
            predictions = model(input_eval, decoder_input)
            predictions = predictions[:, -1, :]
            predicted_id = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1).item()
            input_eval = torch.cat([input_eval, torch.tensor([[predicted_id]], dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')], dim=-1)
            decoder_input = torch.cat([decoder_input, torch.tensor([[predicted_id]], dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')], dim=-1)
            text_generated.append(id2char[predicted_id])

    return start_string + ''.join(text_generated)

print(generate_text(model, start_string="田青文接过羽箭，只看了一眼，"))
