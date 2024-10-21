import torch 
import torch.nn as nn 

class LuongAttention(nn.Module):
    def __init__(self, ):
        super(LuongAttention, self).__init__()
    
    def forward(self, decoder_state, encoder_hiddens):
        batch_size, encoder_sequence_length, encoder_hidden_dim = encoder_hiddens.size()
        attention_score = torch.zeros(batch_size, encoder_sequence_length)
        s_t = decoder_state
        
        for t in range(encoder_sequence_length):
            h_t = encoder_hiddens[:, t]
            attention_score[:, t] = torch.sum(s_t * h_t)

        attention_distribution = torch.softmax(attention_score, dim = 1)

        context_vector = torch.zeros(batch_size, encoder_hidden_dim)
        
        for t in range(encoder_sequence_length):
            context_vector += attention_distribution[:, t].unsqueeze(1) * encoder_hiddens[:, t]

        return context_vector 

class BahdanauAttention(nn.Module):
    def __init__(self, k, h):
        # k : hidden dimension for attention 
        super(BahdanauAttention, self).__init__()
        self.W_a = nn.Linear(k, 1)
        self.W_b = nn.Linear(h, k) # s_t-1: h, 1 -> k, 1
        self.W_c = nn.Linear(h, k) # W_c * H, H: h, L -? k, L 
    
    def forward(self, decoder_state, encoder_hiddens):
        batch_size, encoder_sequence_length, encoder_hidden_dim = encoder_hiddens.size()
        
        attention_score = self.W_a(torch.tanh(self.W_b(decoder_state) + self.W_c(encoder_hiddens)))
        attention_distribution = torch.softmax(attention_score, dim = 1)

        context_vector = torch.zeros(batch_size, encoder_hidden_dim)
        
        for t in range(encoder_sequence_length):
            context_vector += attention_distribution[:, t] * encoder_hiddens[:, t]

        return context_vector 