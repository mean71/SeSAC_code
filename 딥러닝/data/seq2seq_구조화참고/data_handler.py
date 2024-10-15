import random
import torch
from torch.utils.data import DataLoader, random_split
from collections import Counter

# Vocabulary class to handle token-index mapping
class Vocabulary:
    """
    A class to handle the mapping between words and their corresponding indices. This class also handles special tokens
    like PAD, SOS, EOS, and OOV which are essential in sequence modeling for padding, indicating start, end of sequences, 
    and out-of-vocabulary tokens.

    Attributes:
        PAD (str): Padding token.
        SOS (str): Start-of-sequence token.
        EOS (str): End-of-sequence token.
        OOV (str): Out-of-vocabulary token.
        pad_idx (int): Index of the padding token.
        sos_idx (int): Index of the start-of-sequence token.
        eos_idx (int): Index of the end-of-sequence token.
        oov_idx (int): Index of the out-of-vocabulary token.
        word2index (dict): Mapping of words to indices.
        index2word (dict): Mapping of indices to words.
        word_count (dict): Count of each word's occurrences.
        n_words (int): Total number of words including special tokens.
    """
    PAD = '[PAD]'
    SOS = '[SOS]'
    EOS = '[EOS]'
    OOV = '[OOV]'
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2
    oov_idx = 3
    SPECIAL_TOKENS = [PAD, SOS, EOS, OOV]

    def __init__(self, word_count_threshold = 0):
        """
        Initializes the Vocabulary class, adding special tokens and initializing the word-index mappings.
        
        Args:
            word_count_threshold (float): The threshold for filtering OOV words. 
        """
        self.word2index = {}
        self.index2word = {}
        self.word_count = {}
        self.n_words = 0  # Count PAD, SOS, EOS, OOV tokens
        self.threshold = word_count_threshold

        # Special tokens
        self.pad_idx = Vocabulary.pad_idx
        self.sos_idx = Vocabulary.sos_idx
        self.eos_idx = Vocabulary.eos_idx
        self.oov_idx = Vocabulary.oov_idx

        # Initialize the special tokens
        self.add_word(Vocabulary.PAD)
        self.add_word(Vocabulary.SOS)
        self.add_word(Vocabulary.EOS)
        self.add_word(Vocabulary.OOV)

    def add_word(self, word):
        """
        Adds a word to the vocabulary if it doesn't exist, or increments its count if it does.

        Args:
            word (str): The word to add.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word_count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words +=  1
        else:
            self.word_count[word] +=  1

    def add_sentence(self, sentence):
        """
        Adds each word in a sentence to the vocabulary.

        Args:
            sentence (list of str): List of words to add.
        """
        for word in sentence:
            self.add_word(word)

    def word_to_index(self, word):
        """
        Maps a word to its corresponding index. If the word is not found, it returns the index for OOV.

        Args:
            word (str): The word to convert.

        Returns:
            int: Index of the word, or the OOV index if the word is not in the vocabulary.
        """
        return self.word2index.get(word, self.oov_idx)

def indices_to_one_hot(batch, vocab_size):
    """
    Converts a batch of indices to one-hot encoded vectors.

    Args:
        batch (torch.Tensor): Input tensor of indices with shape [batch_size, seq_length].
        vocab_size (int): Size of the vocabulary for the one-hot encoding.

    Returns:
        torch.Tensor: One-hot encoded tensor with shape [batch_size, seq_length, vocab_size].
    """
    assert batch.dim() == 2, f"Input batch should have 2 dimensions, but got {batch.dim()}"
    batch_size, seq_length = batch.size()
    
    one_hot = torch.zeros(batch_size, seq_length, vocab_size).to(batch.device)
    one_hot.scatter_(2, batch.unsqueeze(2), 1)  # [batch_size, seq_length, vocab_size]
    
    assert one_hot.size() == (batch_size, seq_length, vocab_size), f"Output should have shape {(batch_size, seq_length, vocab_size)}, but got {one_hot.size()}"
    
    return one_hot

def generate_target_sequence(input_seq, length, lookback = 3, mod = 10):
    res = []
    
    for idx in range(length):
        if idx < lookback:
            res.append(sum(input_seq[-lookback+idx:] + res[:idx]) % mod)
        else:
            res.append(sum(res[-lookback:]) % mod)
    
    return res 

def create_integer_sequence_dataset(num_samples = 1000, max_input_len = 10, max_target_len = 15, vocab_size = 20, batch_size = 32, train_valid_test = (0.8, 0.1, 0.1)):
    """
    Creates a dataset of integer sequences, where input sequences are randomly generated and their targets are sorted sequences.

    Args:
        num_samples (int): The number of samples to generate in the dataset.
        max_input_len (int): The maximum length of input sequences in the dataset.
        max_target_len (int): The maximum length of target sequences in the dataset.
        vocab_size (int): The size of the vocabulary.
        batch_size (int): The batch size for the DataLoader.
        train_valid_test (tuple): Proportions for splitting the dataset into training, validation, and testing sets.

    Returns:
        tuple: A tuple containing the list of DataLoader objects for training, validation, and testing sets, 
               and the associated Vocabulary object.
    """
    vocab = Vocabulary()
    no_special_token = len(Vocabulary.SPECIAL_TOKENS)
    for i in range(no_special_token, vocab_size):  
        vocab.add_word(str(i))

    data = []
    for _ in range(num_samples):
        input_len = random.randint(3, max_input_len)
        
        input_seq = [random.randint(no_special_token, vocab_size - 1) for _ in range(input_len)]
        target_seq = input_seq 
        # target_seq = generate_target_sequence(input_seq, max_target_len)
        
        # input_seq = [(no_special_token + i) % vocab_size for i in range(input_len)]
        # target_seq = [(input_seq[-1] + i) % vocab_size for i in range(len(input_seq))]

        data.append((input_seq, target_seq))

    # Split dataset
    lengths = [int(num_samples * ratio) for ratio in train_valid_test]
    lengths[-1] = num_samples - sum(lengths[:-1])  # Adjust the last set size to account for rounding

    datasets = random_split(data, lengths)
    dataloaders = [DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn) for dataset in datasets]

    return dataloaders, vocab


def create_lang_pair(data_file, batch_size = 32, train_valid_test = (0.8, 0.1, 0.1)):
    source_vocab = Vocabulary()
    target_vocab = Vocabulary()

    text = open(data_file, 'r', encoding = 'utf-8').read() 
    data = []
    num_samples = 0
    for line in text.split('\n'):
        line = line.strip()
        try:
            source, target, _ = line.split('\t')
        except ValueError:
            try:
                source, target = line.split('\t') 
            except ValueError:
                continue 

        source = source.strip().split()
        target = target.strip().split()
        source_vocab.add_sentence(source)
        target_vocab.add_sentence(target)

        data.append((source, target))
        num_samples += 1

    lengths = [int(num_samples * ratio) for ratio in train_valid_test]
    lengths[-1] = num_samples - sum(lengths[:-1])  # Adjust the last set size to account for rounding

    datasets = random_split(data, lengths)
    dataloaders = [DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda x: collate_fn_language(x, source_vocab, target_vocab)) for dataset in datasets]

    return dataloaders, source_vocab, target_vocab

def collate_fn(batch):
    """
    Custom collate function for batching integer sequence data.
    
    Args:
        batch (list of tuple): List of input-output sequence pairs where input is an unsorted sequence of integers, and target is the sorted sequence.

    Returns:
        tuple: Padded input and target tensors of shape [batch_size, max_seq_len].
    """
    input_seqs = [item[0] for item in batch]
    target_seqs = [item[1] for item in batch]

    # Add EOS token to sequences
    input_seqs = [seq + [Vocabulary.eos_idx] for seq in input_seqs]
    target_seqs = [[Vocabulary.sos_idx] + seq + [Vocabulary.eos_idx] for seq in target_seqs]

    # Get max sequence lengths
    input_max_len = max([len(seq) for seq in input_seqs])
    target_max_len = max([len(seq) for seq in target_seqs])

    # Pad sequences
    input_padded = []
    for seq in input_seqs:
        seq = seq + [Vocabulary.pad_idx] * (input_max_len - len(seq))
        input_padded.append(seq)

    target_padded = []
    for seq in target_seqs:
        seq = seq + [Vocabulary.pad_idx] * (target_max_len - len(seq))
        target_padded.append(seq)

    # Convert to tensors
    input_padded = torch.tensor(input_padded, dtype = torch.long)  # [batch_size, max_seq_len]
    target_padded = torch.tensor(target_padded, dtype = torch.long)  # [batch_size, max_seq_len]

    return input_padded, target_padded

def collate_fn_language(batch, source_vocab, target_vocab):
    """
    Custom collate function for batching language sequence data.

    Args:
        batch (list of tuple): List of input-output sequence pairs where input is an unsorted sequence of words, 
                               and target is the reversed sequence.
        vocab (Vocabulary): Vocabulary object for word-to-index conversions.

    Returns:
        tuple: Padded input and target tensors of shape [batch_size, max_seq_len].
    """
    input_seqs = [item[0] for item in batch]
    target_seqs = [item[1] for item in batch]

    # Convert words to indices and add EOS token
    input_seqs_indices = []
    for seq in input_seqs:
        seq_indices = [source_vocab.word_to_index(word) for word in seq] + [source_vocab.eos_idx]
        input_seqs_indices.append(seq_indices)

    target_seqs_indices = []
    for seq in target_seqs:
        seq_indices = [target_vocab.sos_idx] + [target_vocab.word_to_index(word) for word in seq] + [target_vocab.eos_idx]
        target_seqs_indices.append(seq_indices)

    # Get max sequence lengths
    input_max_len = max([len(seq) for seq in input_seqs_indices])
    target_max_len = max([len(seq) for seq in target_seqs_indices])

    # Pad sequences
    input_padded = []
    for seq in input_seqs_indices:
        seq = seq + [source_vocab.pad_idx] * (input_max_len - len(seq))
        input_padded.append(seq)

    target_padded = []
    for seq in target_seqs_indices:
        seq = seq + [target_vocab.pad_idx] * (target_max_len - len(seq))
        target_padded.append(seq)

    # Convert to tensors
    input_padded = torch.tensor(input_padded, dtype = torch.long)  # [batch_size, max_seq_len]
    target_padded = torch.tensor(target_padded, dtype = torch.long)  # [batch_size, max_seq_len]

    return input_padded, target_padded

if __name__ == '__main__':
    # (train, valid, test), vocab = create_integer_sequence_dataset()

    # for x, y in train:
    #     print(x[0], y[0])
    #     break 

    (train, valid, test), source_vocab, target_vocab = create_lang_pair('kor.txt')

    for x, y in train:
        print(x[0])
        print(y[0])
        break 