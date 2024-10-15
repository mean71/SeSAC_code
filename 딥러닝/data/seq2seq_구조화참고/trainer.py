import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader

from data_handler import Vocabulary
from metrics import bleu 

class Trainer:
    """
    Trainer class to handle training and evaluation of PyTorch models.

    Capabilities:
    - Accepts any model inheriting from nn.Module.
    - Configurable training scenarios via hyperparameters.
    - Plots loss graphs for training and validation sets during training.
    - Evaluates model during training, logs performance, and keeps track of the best model.
    """

    def __init__(self, model, device, criterion, optimizer, max_target_len = 100, save_dir = 'models'):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to train.
            device (torch.device): Device to run the model on.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            save_dir (str, optional): Directory to save the best model.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok = True)
        self.max_target_len = max_target_len

        # For tracking training progress
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train(self, train_loader, valid_loader = None, num_epochs = 10, print_every = 1, evaluate_every = 1, evaluate_metric = 'accuracy'):
        """
        Trains the model.

        Args:
            train_loader (DataLoader): Training data loader.
            valid_loader (DataLoader, optional): Validation data loader.
            num_epochs (int, optional): Number of epochs.
            print_every (int, optional): Frequency of printing training status.
            evaluate_every (int, optional): Frequency of evaluation on validation set.
        """
        val_counter = 0
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_train_loss = 0

            for batch_idx, (source, target) in enumerate(train_loader):
                source = source.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(source, target)

                # Flatten the output and target tensors
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                target = target.reshape(-1)

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Print training status
            if epoch % print_every == 0:
                print(f'Epoch {epoch}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

            # Evaluate on validation set
            if valid_loader and epoch % evaluate_every == 0:
                val_loss, val_metric = self.evaluate(valid_loader, evaluate_metric = evaluate_metric)
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_metric)

                # Save the best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    torch.save(self.best_model_state, os.path.join(self.save_dir, 'best_model.pth'))
                    print(f'Best model saved with validation loss: {val_loss:.4f}, validation {evaluate_metric}: {val_metric:.4f}')
                    val_counter = 0
                else:
                    val_counter += 1 

            if val_counter > 10:
                self.plot_losses()
                return         

        # Plot losses
        self.plot_losses()

    def evaluate(self, valid_loader, evaluate_metric = 'accuracy'):
        """
        Evaluates the model on the validation set.

        Args:
            valid_loader (DataLoader): Validation data loader.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        epoch_val_loss = 0

        if evaluate_metric == 'accuracy':
            correct = 0 
            total = 0
        elif evaluate_metric == 'BLEU':
            bleu_sum = 0
        
        with torch.no_grad():
            for source, target in valid_loader:
                source = source.to(self.device)
                target = target.to(self.device)

                output = self.model(source, target, teacher_forcing_ratio = 0)
                output_dim = output.shape[-1]
                
                if evaluate_metric == 'accuracy':
                    output = output[:, 1:].reshape(-1, output_dim)
                    target = target[:, 1:].reshape(-1)
                    pred = torch.argmax(output, dim = -1)
                    correct += torch.sum((pred == target).float())
                    total += target.numel()
                elif evaluate_metric == 'BLEU':
                    pred = torch.argmax(output, dim = -1)
                    bleu_sum = sum([bleu(pred_sent, target_sent) for pred_sent, target_sent in zip(pred, target)]) / len(target)
                    output = output[:, 1:].reshape(-1, output_dim)
                    target = target[:, 1:].reshape(-1)
                loss = self.criterion(output, target)
                epoch_val_loss += loss.item()
        
        if evaluate_metric == 'accuracy':
            metric = correct / total
        elif evaluate_metric == 'BLEU':
            metric = bleu_sum / len(valid_loader)
        
        avg_val_loss = epoch_val_loss / len(valid_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation {evaluate_metric}: {metric:.4f}')

        return avg_val_loss, metric 

    def inference(self, input_sequence, vocab):
        """
        Performs inference using the trained model.

        Args:
            input_sequence (list): Input sequence as a list of token indices.

        Returns:
            list: Output sequence as a list of tokens.
        """
        self.model.eval()
        source = input_sequence
        source = torch.tensor(source, dtype = torch.long).unsqueeze(0).to(self.device)  # [1, seq_len]

        # Encode
        encoder_state = self.model.encoder(source)

        # Prepare decoder input (<SOS> token)
        input_token = torch.tensor([Vocabulary.sos_idx], device = self.device)

        outputs = []

        hidden = encoder_state.hidden
        cell = encoder_state.cell

        for _ in range(self.max_target_len):
            x_t = self.model.decoder.embedding(input_token)  # [1, embedding_dim]

            if self.model.decoder.model_type == 'rnn':
                hidden = self.model.decoder.cell(x_t, hidden)
                output = self.model.decoder.out(hidden)
            elif self.model.decoder.model_type == 'lstm':
                hidden, cell = self.model.decoder.cell(x_t, hidden, cell)
                output = self.model.decoder.out(hidden)

            top1 = output.argmax(1)
            outputs.append(top1.item())
            input_token = top1

            # Stop if EOS token is generated
            if top1.item() == Vocabulary.eos_idx:
                break

        # Convert output indices to words
        output_sequence = [vocab.index2word[idx] if idx in vocab.index2word else vocab.OOV for idx in outputs ]#  if idx not in range(len(Vocabulary.SPECIAL_TOKENS))]
        return output_sequence

    def plot_losses(self):
        """
        Plots the training and validation losses.
        """
        plt.figure(figsize = (10, 5))
        plt.plot(self.train_losses, label = 'Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label = 'Validation Loss')
            plt.plot(self.val_metrics, label = 'Validation Score')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Graph')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/loss.png')

    def load_best_model(self):
        """
        Loads the best model saved during training.
        """
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print('Best model loaded.')
        else:
            print('No best model found. Please train the model first.')

    def save_checkpoint(self, epoch, filename = 'checkpoint.pth'):
        """
        Saves a checkpoint of the model.

        Args:
            epoch (int): Current epoch.
            filename (str, optional): Filename for the checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        print(f'Checkpoint saved at epoch {epoch}.')

    def load_checkpoint(self, filename = 'checkpoint.pth'):
        """
        Loads a checkpoint of the model.

        Args:
            filename (str, optional): Filename for the checkpoint.
        """
        checkpoint = torch.load(os.path.join(self.save_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_model_state = checkpoint['best_model_state']
        print(f'Checkpoint loaded from epoch {checkpoint["epoch"]}.')
