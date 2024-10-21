import torch
import torch.nn as nn

class RNNCellManual(nn.Module):
    """
    = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = == = 
    Manual Recurrent Neural Network (RNN) Cell
    = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = == = 

    Architecture:
        This RNN cell operates by applying a linear transformation on the input and previous hidden state, 
        followed by a Tanh nonlinearity, which introduces nonlinearity into the hidden state transition.
        
        The theoretical background follows the classic RNN architecture:
        
        Formula:
        h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
        
        Here, `W_ih` and `W_hh` are the weight matrices for the input and hidden states, respectively, and 
        `b_h` is the bias term for the hidden state. The Tanh activation function ensures that the values 
        of the hidden state remain between -1 and 1, which is crucial for gradient flow during backpropagation 
        through time (BPTT).
        
        The class manually implements this architecture without using PyTorch's built-in RNN modules.
        
    Attributes:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden features (i.e., the size of the hidden state).
    """
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the RNN cell with input and hidden dimensions.

        Args:
            input_dim (int): Size of the input feature vector.
            hidden_dim (int): Size of the hidden state.

        """
        super(RNNCellManual, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_ih = nn.Linear(input_dim, hidden_dim)  # Input to hidden
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)  # Hidden to hidden

    def forward(self, x_t, h_t):
        """
        Forward pass for the manual RNN cell.
        
        Args:
            x_t (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            h_t (torch.Tensor): Previous hidden state tensor of shape [batch_size, hidden_dim].

        Returns:
            torch.Tensor: Updated hidden state of shape [batch_size, hidden_dim].
        
        """
        # Assertion to check input dimensions
        batch_size = x_t.shape[0]
        assert x_t.shape[1] == self.input_dim, f"Expected input dimension {self.input_dim}, got {x_t.shape[1]}"
        assert h_t.shape[1] == self.hidden_dim, f"Expected hidden dimension {self.hidden_dim}, got {h_t.shape[1]}"

        # Perform linear transformation and Tanh activation
        h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_t))  # [batch_size, hidden_dim]

        # Assertion to check output dimensions
        assert h_t.shape == (batch_size, self.hidden_dim), f"Expected output shape [{x_t.shape[0]}, {self.hidden_dim}], got {h_t.shape}"

        return h_t


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
        """
        Initialize the LSTM cell with input and hidden dimensions.
        
        Args:
            input_dim (int): Size of the input feature vector.
            hidden_dim (int): Size of the hidden and cell states.
        """
        super(LSTMCellManual, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Input gate components
        self.W_ii = nn.Linear(input_dim, hidden_dim)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim)
        # Forget gate components
        self.W_if = nn.Linear(input_dim, hidden_dim)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim)
        # Cell gate components
        self.W_ig = nn.Linear(input_dim, hidden_dim)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim)
        # Output gate components
        self.W_io = nn.Linear(input_dim, hidden_dim)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, h_t, c_t):
        """
        Forward pass for the manual LSTM cell.
        
        Args:
            x_t (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            h_t (torch.Tensor): Previous hidden state tensor of shape [batch_size, hidden_dim].
            c_t (torch.Tensor): Previous cell state tensor of shape [batch_size, hidden_dim].

        Returns:
            torch.Tensor: Updated hidden state tensor of shape [batch_size, hidden_dim].
            torch.Tensor: Updated cell state tensor of shape [batch_size, hidden_dim].
        """
        # Assertions to check input dimensions
        
        batch_size = x_t.shape[0]
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_dim)
        assert x_t.shape[1] == self.input_dim, f"Expected input dimension {self.input_dim}, got {x_t.shape[1]}"
        assert h_t.shape[1] == self.hidden_dim, f"Expected hidden dimension {self.hidden_dim}, got {h_t.shape[1]}"
        assert c_t.shape[1] == self.hidden_dim, f"Expected cell state dimension {self.hidden_dim}, got {c_t.shape[1]}"

        # Input gate computation
        i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h_t))  # [batch_size, hidden_dim]
        # Forget gate computation
        f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h_t))  # [batch_size, hidden_dim]
        # Cell gate computation (proposed new state)
        g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h_t))     # [batch_size, hidden_dim]
        # Output gate computation
        o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h_t))  # [batch_size, hidden_dim]

        # Cell state update
        c_t = f_t * c_t + i_t * g_t                            # [batch_size, hidden_dim]

        # Hidden state update
        h_t = o_t * torch.tanh(c_t)                            # [batch_size, hidden_dim]

        # Assertions to check output dimensions
        assert h_t.shape == (batch_size, self.hidden_dim), f"Expected output hidden state shape [{x_t.shape[0]}, {self.hidden_dim}], got {h_t.shape}"
        assert c_t.shape == (batch_size, self.hidden_dim), f"Expected output cell state shape [{x_t.shape[0]}, {self.hidden_dim}], got {c_t.shape}"

        return h_t, c_t
