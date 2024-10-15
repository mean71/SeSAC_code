import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data_handler import Vocabulary
from rnn_cell import RNNCellManual, LSTMCellManual

class EncoderState:
    """Abstraction for the encoder outputs to be passed to the decoder.
    This class encapsulates all necessary information from the encoder
    to the decoder, allowing flexibility in encoder outputs.
    """
    def __init__(self, hidden, **kargs):
        self.hidden = hidden
        self.extra_info = kargs  # For any additional data
        for k, v in kargs.items():
            exec(f'self.{k} = v')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, model_type = 'rnn'):
        super(Encoder, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == 'rnn':
            self.cell = RNNCellManual(embedding_dim, hidden_size)
        elif model_type == 'lstm':
            self.cell = LSTMCellManual(embedding_dim, hidden_size)
        else:
            raise ValueError('Invalid model type')

    def forward(self, source):
        batch_size, seq_len = source.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(source.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(source.device) if self.model_type == 'lstm' else None

        # Embed the source sequence
        embedded = self.embedding(source)  # [batch_size, seq_len, embedding_dim]

        # Encode the source sequence
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # [batch_size, embedding_dim]

            if self.model_type == 'rnn':
                h_t = self.cell(x_t, h_t)
            elif self.model_type == 'lstm':
                h_t, c_t = self.cell(x_t, h_t, c_t)

        # Return an EncoderState object
        return EncoderState(hidden = h_t, cell = c_t)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, model_type='rnn'):
        super(Decoder, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == 'rnn':
            self.cell = RNNCellManual(embedding_dim, hidden_size)
        elif model_type == 'lstm':
            self.cell = LSTMCellManual(embedding_dim, hidden_size)
        else:
            raise ValueError('Invalid model type')

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, target, encoder_state, teacher_forcing_ratio = 0.5, eos_idx = Vocabulary.eos_idx):
        hidden = encoder_state.hidden
        cell = encoder_state.cell
        batch_size, trg_len = target.size()
        outputs = torch.zeros(batch_size, trg_len, self.vocab_size).to(target.device)

        # First input to the decoder is the <SOS> tokens
        input = torch.tensor([[Vocabulary.sos_idx] for _ in range(batch_size)], dtype = torch.long).to(target.device) # [batch_size, 1]

        for t in range(trg_len):
            x_t = self.embedding(input).unsqueeze(1)  # [batch_size, embedding_dim]
            print(x_t.shape)
            if self.model_type == 'rnn':
                hidden = self.cell(x_t, hidden)
            elif self.model_type == 'lstm':
                hidden, cell = self.cell(x_t, hidden, cell)
            
            output = self.out(hidden)
            outputs[:, t] = output

            # Decide whether to do teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # Get the predicted word

            input = target[:, t+1] if teacher_force and t < trg_len - 1 else top1

            # Optional: Break early if all sequences in batch have generated EOS
            if not teacher_force and (top1 == eos_idx).all():  
                break

        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio = 0.5):
        # Encode
        encoder_state = self.encoder(source)

        # Decode
        outputs = self.decoder(target, encoder_state, teacher_forcing_ratio)

        return outputs

