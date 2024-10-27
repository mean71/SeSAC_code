import torch
import torch.nn as nn
# 배치 디멘전이 0이라고가정해서 눈치껏 알아야 한다 그게 미니배치에 편하다 무슨소리지
# 배치 퍼스트?  그런 가정을 암묵적으로 담아서 나쁜 코드이다
class LayerNormalization(nn.Module): # 많이 나와서 중요하다 마지막에 잘 알아둬봐라 # 표준정규분포로 normalization해주는게 이것의 요지다?
    def __init__(
        self,
        input_dim,
        eps: float = 1e-6,
    ):
        self.gamma = nn.Parameter(torch.ones(input_dim))# 트랜프포머할때 파라미터쓰는게 정석이고 음 어... 이걸 뭐 어쩌다고요...? # 그냥 받아들이라
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.eps = eps
        # self.gamma = torch.tensor(..., requires_grad = True)
    
    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        mean = torch.mean(x)
        std = torch.std(x)
        
        return self.gamma * (x-mean) / (std + self.eps) + self.beta

class PositnionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int
    ):
        super(PositnionalEncoding, self).__init__()
        positnional_encoding = torch.zeros(max_len, embedding_dim)
        
        position = torch.arange(0, max_len)

class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attention_head_dim: int,
    ):
        self.W_q: nn.Linear = nn.Linear(embedding_dim, attention_head_dim)
        self.W_k: nn.Linear = nn.Linear(embedding_dim, attention_head_dim)
        self.W_v: nn.Linear = nn.Linear(embedding_dim, attention_head_dim)
        
        self.softmax: nn.SoftMax = nn.SoftMax(dim = -1)
        self.attention_head_dim = attention_head_dim
    
    def forward(
        self,
        x: torch.tensor
    ):
        Q: torch.tensor = self.W_q(x)
        K: torch.tensor = self.W_k(x)
        V: torch.tensor = self.W_v(x)
        
        score

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ):
        super(MultiheadSelfAttention, self).__init__()
        self.heads: List[SelfAttention] = [SelfAttention(embedding_dim)
                                           for _ in range(num_heads)]
        self.layer: nn.Linear = nn.Linear(num_heads * embedding_dim, embedding_dim)
    
    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        x = torch.cat([head(x) for head in self.head], dim = 1)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ):
        super(TransformerEncoderLayer, self).__init__()
    
    def forward(
        self,
        x: torch.tensor,
    ):
        pass

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_heads: int,
        attentnion_head_dim: int,
    ):
        super(TransformerEncoder, self).__init__()
        
        self.layers: List[TransformerEncoder] = [TransformerEncoderLayer(embedding_dim, num_heads, attentnion_head_dim)
                                                 for _ in range(num_layers)]
        self.positional_encoding: PositnionalEncoding = PositnionalEncoding(embedding_dim)
        
        def forward(
            self.\
        )

class Transformer(nn.Module):
    
    # 마스킹 인코더와 디코더 사이 어탠션
    # 이거를 가지고 시퀀2시퀀으 붙여서 돌려야한다
    def __init__(
        self,
        num_heads: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int,
        num_layers: int,
    ):
        super(Transformer, self).__init__()
        self.encoding_embedding: nn.Embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.decoding_embedding: nn.Embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.encoder: TransformerEncoder = TransformerEncoder(num_layers, embedding_dim, num_heads)
        self.decoder: TransformerDecoder = TransformerDecoder(num_layers, embedding_dim, num_heads)

    def forward(
        self,
        src: torch.tensor,
        tgt: torch.tensor,
    ) -> torch.tensor:
        src_embedding: torch.tensor = self.encoder_embedding(src)
        encoder_output: torch.tensor = self.encoder(src_embedding)