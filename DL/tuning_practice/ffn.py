import matplotlib.pyplot as plt
import os
import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from util import chain 
from tuning import BatchNormalization

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(len(alphabets) * max_length, hidden_size)
        self.bn1 = BatchNormalization(hidden_size)  # 배치 정규화
        self.fc2 = nn.Linear(hidden_size, len(languages))

    def forward(self, x):
        # x: (batch_size, max_length, len(alphabets) : 32, 12, 57) -> (32, 12*57)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim = -1)

        return x # (batch_size, len(languages) : 32, 18)

    def train_model(self, train_data, valid_data, epochs = 20, learning_rate = 0.001, print_every = 1000):
        criterion = F.nll_loss
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)

        step = 0
        train_loss_history = []
        valid_loss_history = []

        train_log = {}

        for epoch in range(epochs):
            for x, y in train_data:
                step += 1
                y_pred = self(x)
                loss = criterion(y_pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mean_loss = torch.mean(loss).item()

                if step % print_every == 0 or step == 1:
                    train_loss_history.append(mean_loss)
                    valid_loss, valid_acc = self.evaluate(valid_data)
                    valid_loss_history.append(valid_loss)
                    print(f'[Epoch {epoch}, Step {step}] train loss: {mean_loss}, valid loss: {valid_loss}, valid_acc: {valid_acc}')
                    # 저장
                    checkpoint_path = os.path.join(checkpoint_dir, f'feedforward_{step}.chkpts')
                    torch.save(self, checkpoint_path)
                    print(f'saved model to checkpoints/feedforward_{step}.chkpts')
                    # 기록
                    train_log[f'checkpoints/feedforward_{step}.chkpts'] = [valid_loss, valid_acc]
        # 로그 저장 pichkle.dump( )
        pickle.dump(train_log, open('checkpoints/train_log.dict', 'wb+'))

        return train_loss_history, valid_loss_history

    def evaluate(self, data):
        self.eval()
        criterion = F.nll_loss

        correct, total = 0, 0
        loss_list = []
        with torch.no_grad():
            for x, y in data:
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss_list.append(torch.mean(loss).item())
                correct += torch.sum((torch.argmax(y_pred, dim = 1) == y).float())
                total += y.size(0)
            return sum(loss_list) / len(loss_list), correct / total

if __name__ == '__main__':
    import config
    from data_handler import generate_dataset, modify_dataset_for_ffn, plot_loss_history

    train_dataset, valid_dataset, test_dataset, alphabets, max_length, languages = generate_dataset()
    dataset = chain(train_dataset, valid_dataset, test_dataset)

    train_data, valid_data, test_data = modify_dataset_for_ffn(dataset)

    model = FeedForwardNetwork(32)
    loss, acc = model.evaluate(train_data)

    train_loss_history, valid_loss_history = model.train_model(train_data, valid_data)
    
    plot_loss_history(train_loss_history, valid_loss_history)
    print("finish")