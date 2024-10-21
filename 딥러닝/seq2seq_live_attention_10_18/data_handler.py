import torch 
from torch.utils.data import DataLoader, random_split 

from collections import defaultdict
from debug_shell import debug_shell 

class Vocabulary:
    PAD = '[PAD]'
    SOS = '[SOS]'
    EOS = '[EOS]'
    OOV = '[OOV]'
    SPECIAL_TOKENS = [PAD, SOS, EOS, OOV] 
    PAD_IDX = 0
    SOS_IDX = 1 
    EOS_IDX = 2 
    OOV_IDX = 3

    def __init__(self, word_count, coverage = 0.999):
        """Accept word_count dictionary having word as key, and frequency as value.
        """
        word_freq_list = []
        total = 0
        
        for word, freq in word_count.items():
            word_freq_list.append((word, freq)) 
            total += freq 

        word_freq_list = sorted(word_freq_list, key = lambda x: x[1], reverse = True)
         
        word2idx = {}
        idx2word = {}
        s = 0
        
        for idx, (word, freq) in enumerate([(e, 0) for e in Vocabulary.SPECIAL_TOKENS] + word_freq_list):
            s += freq 
            if s > coverage * total:
                break 
            word2idx[word] = idx 
            idx2word[idx] = word 
        
        self.word2idx = word2idx 
        self.idx2word = idx2word 
        self.vocab_size = len(word2idx)

    def word_to_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return Vocabulary.OOV_IDX

def parse_file(file_path, 
        train_valid_test_ratio = (0.8, 0.1, 0.1), 
        batch_size = 32, ):
    f = open(file_path, 'r', encoding = 'utf-8')
    data = []

    source_word_count = defaultdict(int)
    target_word_count = defaultdict(int)
    
    for line in f.readlines():
        line = line.strip()
        source, target, etc = line.split('\t')

        source = source.split()
        for source_token in source:
            source_word_count[source_token] += 1 

        target = target.split()
        for target_token in target:
            target_word_count[target_token] += 1

        data.append((source, target))

    source_vocab = Vocabulary(source_word_count)
    target_vocab = Vocabulary(target_word_count)

    for idx, (source, target) in enumerate(data):
        data[idx] = (
            list(map(source_vocab.word_to_index, source)), 
            list(map(target_vocab.word_to_index, target)), 
        )

    lengths = [int(len(data) * ratio) for ratio in train_valid_test_ratio]
    lengths[-1] = len(data) - sum(lengths[:-1])
    datasets = random_split(data, lengths)
    dataloaders = [\
        DataLoader(dataset, 
                batch_size = batch_size, 
                shuffle = True, 
                collate_fn = lambda x: preprocessing(x, source_vocab, target_vocab)) 
    for dataset in datasets]

    return dataloaders, source_vocab, target_vocab 

def preprocessing(batch, source_vocab, target_vocab):
    sources = [e[0] for e in batch] 
    targets = [e[1] for e in batch] 

    source_seqs = []
    target_seqs = []

    for source_seq in sources:
        source_seqs.append(source_seq + [source_vocab.EOS_IDX])
    
    for target_seq in targets:
        target_seqs.append([target_vocab.SOS_IDX] + target_seq + [target_vocab.EOS_IDX])

    source_max_length = max([len(s) for s in source_seqs]) 
    target_max_length = max([len(s) for s in target_seqs]) 

    for idx, seq in enumerate(source_seqs):
        seq = seq + [source_vocab.PAD_IDX] * (source_max_length - len(seq))
        assert len(seq) == source_max_length, f'Expected to have {source_max_length}, now {len(seq)}'
        source_seqs[idx] = seq 

    for idx, seq in enumerate(target_seqs):
        seq = seq + [target_vocab.PAD_IDX]* (target_max_length - len(seq))
        assert len(seq) == target_max_length, f'Expected to have {target_max_length}, now {len(seq)}'
        target_seqs[idx] = seq 
        
    return torch.tensor(source_seqs), torch.tensor(target_seqs)

if __name__ == '__main__':
    from code import interact 
    batch_size = 32 
    (train, valid, test), source_vocab, target_vocab = parse_file('kor.txt', batch_size = batch_size)
    
    for source_batch, target_batch in train:
        assert source_batch.shape[0] == batch_size
        print(source_batch) 
        
        assert target_batch.shape[0] == batch_size 
        print(target_batch)