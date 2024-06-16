import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

class Seq2SeqModel(nn.Module):
    def __init__(self, hidden_size, hidden_layers, vocab_size):
        super(Seq2SeqModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm_layers = nn.ModuleList([nn.LSTM(hidden_size, hidden_size, batch_first=True) for _ in range(hidden_layers)])
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, states=None, return_state=False):
        x = self.embedding(inputs)
        new_states = []

        if states is None:
            states = [None] * self.hidden_layers

        for i in range(self.hidden_layers):
            x, (state_h, state_c) = self.lstm_layers[i](x, states[i])
            new_states.append((state_h, state_c))

        x = self.dense(x)

        if return_state:
            return x, new_states
        else:
            return x

hidden_size = 128
hidden_layers = 2
vocab_size = len(vocab)
batch_size = 64
time_steps = 30
epochs = 100
learning_rate = 0.005

model = Seq2SeqModel(hidden_size, hidden_layers, vocab_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义数据生成器
data_loader = create_data_loader(numdata, batch_size, time_steps)


class LossHistory:
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, loss):
        self.losses.append(loss)


history = LossHistory()

model.train()
for epoch in range(epochs):
    total_loss = 0
    for x, y in data_loader:
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs.view(-1, vocab_size), y.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    history.on_epoch_end(average_loss)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')

plt.plot(history.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Seq2Seq Model Training Loss')
plt.show()


def generate_text(model, start_string, num_generate=100):
    model.eval()
    input_eval = torch.tensor([char2id[s] for s in start_string], dtype=torch.long).unsqueeze(0)

    text_generated = []
    states = None

    for i in range(num_generate):
        with torch.no_grad():
            outputs, states = model(input_eval, states=states, return_state=True)
            predictions = outputs[:, -1, :]  # 获取最后一个时间步的输出
            predicted_id = torch.multinomial(F.softmax(predictions, dim=-1), num_samples=1).item()

        input_eval = torch.tensor([[predicted_id]], dtype=torch.long)
        text_generated.append(id2char[predicted_id])

    return start_string + ''.join(text_generated)


print(generate_text(model, start_string="田青文接过羽箭，只看了一眼，"))