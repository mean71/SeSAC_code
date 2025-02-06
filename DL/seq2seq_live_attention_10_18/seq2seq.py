import random 
import torch 
import torch.nn as nn 

from attentions import LuongAttention, BahdanauAttention

class EncoderState:
    def __init__(self, **kargs):
        for k, v in kargs.items():
            exec(f'self.{k} = v')

    def initialize(self):
        assert 'model_type' in dir(self)
        return self.model_type.initialize()

class Encoder(nn.Module):
    def __init__(self, source_vocab, embedding_dim, hidden_dim, model_type, ):
        super(Encoder, self).__init__()
        self.source_vocab = source_vocab 
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim 
        self.model_type = model_type

        self.embedding = nn.Embedding(source_vocab.vocab_size, embedding_dim)
        self.cell = model_type(embedding_dim, hidden_dim)

    def forward(self, source):
        batch_size, seq_length = source.size()
        hiddens = []

        embedded = self.embedding(source)
        encoder_state = self.cell.initialize(batch_size)

        for t in range(seq_length):
            x_t = embedded[:, t, :]
            if self.model_type == RNNCellManual:
                encoder_state = self.cell(x_t, encoder_state)
                hiddens.append(encoder_state)
            elif self.model_type == LSTMCellManual:
                encoder_state = self.cell(x_t, *encoder_state)
                hiddens.append(encoder_state[0])

        return torch.stack(hiddens, dim = 1)

class Decoder(nn.Module):
    def __init__(self, target_vocab, embedding_dim, hidden_dim, model_type, attention):
        super(Decoder, self).__init__()
        
        self.target_vocab = target_vocab 
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim 
        self.model_type = model_type        

        if attention == LuongAttention:
            self.attention = LuongAttention()
            self.W_c = nn.Linear(target_vocab.vocab_size + hidden_dim, target_vocab.vocab_size)
            self.embedding = nn.Embedding(target_vocab.vocab_size, embedding_dim)
            self.cell = model_type(embedding_dim, hidden_dim) 
        elif attention == BahdanauAttention:
            self.attention = BahdanauAttention()
            self.cell = model_type(embedding_dim + hidden_dim, hidden_im)
        else:
            self.embedding = nn.Embedding(target_vocab.vocab_size, embedding_dim)
            self.cell = model_type(embedding_dim, hidden_dim) 
        self.h2o = nn.Linear(hidden_dim, target_vocab.vocab_size)
            
            

    def forward(self, target, encoder_hiddens,  teacher_forcing_ratio = 0.5):
        encoder_last_state = encoder_hiddens[:, -1]
        
        # target: batch_size, seq_length 
        batch_size, seq_length = target.size()
        
        outputs = []

        input = torch.tensor([self.target_vocab.SOS_IDX for _ in range(batch_size)]) # input: batch_size 
        decoder_state = encoder_last_state 

        for t in range(seq_length):
            # embedded: batch_size * embedding_dim 
            embedded = self.embedding(input)
            if isinstance(self.attention, BahdanauAttention):
                context_vector = self.attention(encoder_hiddens, decoder_state)
                embedded = torch.cat((embedded, context_vector), dim = 1) # embedded: batch_size, hidden_dim + embedding_size 
            
            if self.model_type == RNNCellManual:
                decoder_state = self.cell(embedded, decoder_state)
            elif self.model_type == LSTMCellManual:
                decoder_state = self.cell(embedded, *decoder_state)
            output = self.h2o(decoder_state) 
            
            if isinstance(self.attention, LuongAttention):
                context_vector = self.attention(decoder_state, encoder_hiddens)
                output = torch.cat((output, context_vector), dim = 1) # output: batch_size, (self.output_dim + encoder_hidden_dim)
                output = torch.tanh(self.W_c(output)) # output: batch_size, target.vocab_size 
            
            outputs.append(output) 

            if random.random() < teacher_forcing_ratio and t < seq_length - 1: # do teacher forcing 
                input = target[:, t+1]
            else:
                input = torch.argmax(output, dim = 1)
        # outputs: batch_size, seq_length, vocab_size 
        return torch.stack(outputs, dim = 1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__() 
        self.encoder = encoder 
        self.decoder = decoder 

    def forward(self, source, target):
        encoder_hiddens = self.encoder(source) 
        outputs = self.decoder(target, encoder_hiddens)

        return outputs 

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    from data_handler import parse_file 
    from rnn_cells import RNNCellManual, LSTMCellManual

    embedding_dim = 256
    batch_size = 32
    encoder_model = RNNCellManual
    decoder_model = RNNCellManual
    hidden_dim = 128
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam 
    learning_rate = 0.001 
    num_epochs = 10 

    (train, valid, test), source_vocab, target_vocab = parse_file('kor.txt', batch_size = batch_size)
    encoder = Encoder(source_vocab, embedding_dim, hidden_dim, encoder_model)
    decoder = Decoder(target_vocab, embedding_dim, hidden_dim, decoder_model, attention = LuongAttention)

    model = Seq2Seq(encoder = encoder, 
                    decoder = decoder,)
    
    model.train()
    loss = 0
    loss_history = []
    
    criterion = criterion()
    optimizer = optimizer(model.parameters(), lr = learning_rate)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        for step_idx, (source_batch, target_batch) in enumerate(train):
            optimizer.zero_grad()
            
            pred_batch = model(source_batch, target_batch)
            batch_size, seq_length = target_batch.size()
            
            loss = criterion(pred_batch.view(batch_size * seq_length, -1), target_batch.view(-1))

            loss.backward() 
            optimizer.step()
            
            epoch_loss += loss 

            if step_idx % 1 == 0:
                print(f'[Epoch {epoch}/{num_epochs}] step {step_idx}: loss - {loss}')

        avg_loss = epoch_loss.item() / len(train)
        loss_history.append(avg_loss)
    plt.plot(loss_history)
    plt.show()
            