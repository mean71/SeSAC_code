import torch
import torch.nn as nn 

class RNNCellManual(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCellManual, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim) 

    def forward(self, x_t, h_t):
        """
        Args:
            x_t: batch_size, input_dim 
            h_t: batch_size, hidden_dim 
        Returns:
            h_t: batch_size, hidden_dim 
        """
        batch_size = x_t.size(0) 
        assert x_t.size(1) == self.input_dim, f'Input dimension was expected to be {self.input_dim}, got {x_t.size(1)}'
        assert h_t.size(0) == batch_size, f'0th dimension of h_t is expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'Hidden dimension was expected to be {self.hidden_dim}, got {h_t.size(1)}' 
        
        h_t = torch.tanh(self.i2h(x_t) + self.h2h(h_t))

        assert h_t.size(0) == batch_size, f'0th dimension of output of RNNManualCell is expected to be {batch_size}, got {h_t.size(0)}'
        assert h_t.size(1) == self.hidden_dim, f'1st dimension of output of RNNManualCell is expected to be {self.hidden_dim}, got {h_t.size(1)}'
        
        return h_t 

    def initialize(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)

class LSTMCellManual(nn.Module):
    """
    = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    Manual Long Short-Term Memory (LSTM) Cell
    = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

    Architecture:
        This LSTM cell manually implements the input, forget, and output gates, 
        as well as the cell state computation. The gates regulate information flow 
        through the network, while the cell state allows long-term memory retention.
        
        Formula:
        i_t = sigmoid(W_ii * x_t + W_hi * h_{t-1} + b_i)  # Input gate
        f_t = sigmoid(W_if * x_t + W_hf * h_{t-1} + b_f)  # Forget gate
        g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_g)     # Cell gate (proposed new state)
        o_t = sigmoid(W_io * x_t + W_ho * h_{t-1} + b_o)  # Output gate
        
        c_t = f_t * c_{t-1} + i_t * g_t  # Updated cell state
        h_t = o_t * tanh(c_t)            # Updated hidden state
        
        The LSTM cell architecture allows the model to retain long-term dependencies by 
        controlling what information should be forgotten and what should be passed on. 
        
    Attributes:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden features (i.e., the size of the hidden and cell states).
    """
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCellManual, self).__init__()
        self.i2i = nn.Linear(input_dim, hidden_dim) 
        self.h2i = nn.Linear(hidden_dim, hidden_dim) 
        self.i2f = nn.Linear(input_dim, hidden_dim) 
        self.h2f = nn.Linear(hidden_dim, hidden_dim) 
        self.i2g = nn.Linear(input_dim, hidden_dim) 
        self.h2g = nn.Linear(hidden_dim, hidden_dim) 
        self.i2o = nn.Linear(input_dim, hidden_dim) 
        self.h2o = nn.Linear(hidden_dim, hidden_dim) 

        self.hidden_dim = hidden_dim 
        self.input_dim = input_dim 

    def forward(self, x_t, h_t, c_t):
        batch_size = x_t.size(0) 
        assert x_t.size(1) == self.input_dim 
        
        assert h_t.size(0) == batch_size
        assert h_t.size(1) == self.hidden_dim 
        
        assert c_t.size(0) == batch_size 
        assert c_t.size(1) == self.hidden_dim 

        i_t = torch.sigmoid(self.i2i(x_t) + self.h2i(h_t))
        f_t = torch.sigmoid(self.i2f(x_t) + self.h2f(h_t))
        g_t = torch.tanh(self.i2g(x_t) + self.h2g(h_t))
        i_t = torch.sigmoid(self.i2o(x_t) + self.h2o(h_t))

        c_t = f_t * c_t + i_t * g_t 
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t 

    def initialize(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim), torch.zeros(batch_size, hidden_dim)

