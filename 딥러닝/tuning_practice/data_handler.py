import glob
import matplotlib.pyplot as plt
import os 
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

import config 

def letter2tensor(letter, alphabets, oov = config.OOV):
    res = [0]*len(alphabets)

    idx = alphabets.index(
        letter
        if (letter in alphabets)
        else oov
        )
    
    res[idx] = 1

    return torch.tensor(res)

def word2tensor(word, max_length, alphabets, pad = config.PAD, oov = config.OOV):
    # return torch.tensor with size (max_length, len(alphabets))
    res = torch.zeros(max_length, len(alphabets))

    for idx, char in enumerate(word):
        if idx < max_length:
            res[idx] = letter2tensor(char, alphabets, oov = oov)

    for i in range(max_length - len(word)):
        res[len(word) + i] = letter2tensor(pad, alphabets, oov = oov)

    return res

def determine_alphabets(data, pad = config.PAD, oov = config.OOV, threshold = 0.999):
    # data = list of [name, language_name]
    lst = []
    character_dict = defaultdict(int)

    for name, lang in data:
        for char in name:
            character_dict[char.lower()] += 1

    for k, v in character_dict.items():
        lst.append((k, v))

    lst = sorted(lst, key = lambda x:x[1], reverse = True)
    total_count = sum([e[1] for e in lst])
    s = 0

    alphabets = []

    for k, v in lst:
        s += v
        if s < threshold * total_count:
            alphabets.append(k)

    alphabets.append(pad)
    alphabets.append(oov)

    return alphabets

def determine_max_length(data, threshold = 0.99):
    lst = []
    name_length_dict = defaultdict(int)

    for name, lang in data:
         name_length_dict[len(name)] += 1

    for k, v in name_length_dict.items():
        lst.append((k, v))

    lst = sorted(lst, key = lambda x:x[1], reverse = True)
    total_count = sum([e[1] for e in lst])
    s = 0

    for k, v in lst:
        s += v
        if s > threshold * total_count:
            return k - 1
    # if not, return the maximum value in lst
    return max(lst, key = lambda x:x[0])[0]

def load_file():
    files = glob.glob(config.file_dir)

    assert len(files) == 18

    data = []
    languages = []

    for file in files:
        with open(file) as f:
            names = f.read().strip().split('\n')
        lang = file.split(os.sep)[-1].split('.')[0]

        if lang not in languages:
            languages.append(lang)

        for name in names:
            data.append([name, lang])

    return data, languages

def generate_dataset(batch_size = config.batch_size, pad = config.PAD, oov = config.OOV):
    data, languages = load_file()

    alphabets = determine_alphabets(data, pad = pad, oov = oov)
    max_length = determine_max_length(data)

    for idx, elem in enumerate(data):
        tmp = []
        for char in elem[0]:
            if char.lower() in alphabets:
                tmp.append(char.lower())
            else:
                tmp.append(oov)

        data[idx][0] = word2tensor(tmp, max_length, alphabets, pad = pad, oov = oov)
        data[idx][1] = languages.index(data[idx][1])

    x = [e[0] for e in data]
    y = [torch.tensor(e[1]) for e in data]

    train_x, train_y, valid_x, valid_y, test_x, test_y = split_train_valid_test(x, y)

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)
    valid_x = torch.stack(valid_x)
    valid_y = torch.stack(valid_y)
    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_dataloader, valid_dataloader, test_dataloader, alphabets, max_length, languages

def modify_dataset_for_ffn(dataset):
    x = []
    y = []

    for batch_x, batch_y in dataset:
        for i in range(batch_x.size(0)):
            x.append(batch_x[i].reshape((batch_x.size(1) * batch_x.size(2))))
            y.append(batch_y[i])

    train_x, train_y, valid_x, valid_y, test_x, test_y = split_train_valid_test(x, y)

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)
    valid_x = torch.stack(valid_x)
    valid_y = torch.stack(valid_y)
    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = config.batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = True)

    return train_dataloader, valid_dataloader, test_dataloader

def pick_train_valid_test(train, valid, test):
    assert [train, valid, test] != [0, 0, 0]
    options = [train, valid, test]

    pick = random.choice([0, 1, 2])

    while options[pick] == 0:
        pick = random.choice([0, 1, 2])
    assert options[pick] != 0
    return pick

def split_train_valid_test(x, y, train_valid_test_ratio = (0.7, 0.15, 0.15)):
    # TensorDataset -> TensorDataset, TensorDataset, TensorDataset
    # x, y: list of data
    train_ratio, valid_ratio, test_ratio = train_valid_test_ratio
    y_label_dict = defaultdict(int)
    for y_data in y:
        y_label_dict[y_data.item()] += 1

    no_per_labels = {} # y_label별로 각각 train, valid, test

    for y_label, freq in y_label_dict.items():
        train_size, valid_size, test_size = int(freq * train_ratio), int(freq * valid_ratio), freq - int(freq * train_ratio) - int(freq * valid_ratio)
        no_per_labels[y_label] = [train_size, valid_size, test_size]

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []

    for x_data, y_data in zip(x, y):
        idx = pick_train_valid_test(*no_per_labels[y_data.item()])
        assert no_per_labels[y_data.item()][idx] > 0
        no_per_labels[y_data.item()][idx] -= 1

        if idx == 0:
            train_x.append(x_data)
            train_y.append(y_data)
        elif idx == 1:
            valid_x.append(x_data)
            valid_y.append(y_data)
        elif idx == 2:
            test_x.append(x_data)
            test_y.append(y_data)

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def plot_loss_history(loss_history, other_loss_history = [], save_dir='loss_history_plot'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_count = len([name for name in os.listdir(save_dir) if name.startswith('loss_history')])
    save_path = os.path.join(save_dir, f'loss_history_{file_count + 1}.png')
    
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss')
    if other_loss_history != []:
        plt.plot(range(1, len(other_loss_history) + 1), other_loss_history, label='Validation Loss')
    plt.legend()
    # plt.show()
    plt.savefig(save_path)
    # plt.close()

def tensor2word(t, alphabets):
    # t.shpae: max_length, len(alphabets)
    res = []
    for char_tensor in t:
        char = alphabets[int(torch.argmax(char_tensor).item())]
        res.append(char)

    return res

def idx2lang(idx, languages):
    return languages[idx]

if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset, alphabets, max_length, languages  = generate_dataset()

    for batch_x, batch_y in train_dataset:
        for i in range(batch_x.size(0)):
            print(tensor2word(batch_x[i], alphabets), idx2lang(batch_y[i], languages))
        break